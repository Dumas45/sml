"""Toy Multi-Head Self-Attention trained as a Causal Language Model (GPT-style).

This educational module implements an explicit multi-head causal self-attention network
trained on raw text from Alice in Wonderland to demonstrate:
1. "Why lower triangular causal masks exist": How autoregressive language models prevent
   information leakage from future tokens using additive causal masks (-inf in the upper triangle).
2. "Autoregressive generation with sampling controls": How next-token prediction enables
   dynamic text synthesis with repetition penalties and nucleus (top-p) sampling.
3. "Transformer decoder architecture": How residual streams and LayerNorm eliminate attention
   sink collapse, allowing heads to distribute focus across past context tokens.
4. "Why toy models overfit & anisotropy": How training on minimal text causes contextual
   embeddings to cluster in a narrow cone, and why last-token pooling aligns with causal
   receptive fields compared to bidirectional mean pooling.
5. "Why low frequencies exist in positional encodings": How sinusoidal positional encodings
   scale across short and long sequence horizons.
"""

from collections import namedtuple
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
import torch.nn.functional as F


# =====================================================================
# 1. DATA PREPARATION & TOKENIZATION
# =====================================================================

def load_text(text_path: Path) -> str:
    """Fetch the text from a given file path.

    Args:
        text_path (Path): Path to the text file.

    Returns:
        str: The raw text content of the file.
    """
    return text_path.read_text(encoding="UTF-8")


def tokenize(text: str) -> list[str]:
    """Tokenize input text at word level while preserving basic punctuation.

    Args:
        text (str): Raw string to tokenize.

    Returns:
        list[str]: Extracted list of lowercase tokens and punctuation marks.
    """
    text = text.lower()
    text = text.replace("_", "")
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


class Vocabulary:
    """Vocabulary mapping between unique tokens and numerical indices.

    Attributes:
        pad_token (str): Special token representing padding.
        unk_token (str): Special token representing out-of-vocabulary words.
        special_tokens (list[str]): List of all registered special tokens.
        token2id (dict[str, int]): Mapping from token strings to integer IDs.
        id2token (dict[int, str]): Mapping from integer IDs to token strings.
    """

    def __init__(
        self,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
    ) -> None:
        """Initialize Vocabulary with special tokens.

        Args:
            pad_token (str): Symbol for padding token. Defaults to '[PAD]'.
            unk_token (str): Symbol for unknown token. Defaults to '[UNK]'.
        """
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.special_tokens = [pad_token, unk_token]
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        for st in self.special_tokens:
            self.add_token(st)

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary if not already present.

        Args:
            token (str): Token string to insert.

        Returns:
            int: The integer ID assigned to the token.
        """
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def build_vocab(self, tokens: list[str]) -> None:
        """Populate vocabulary from an iterable collection of tokens.

        Args:
            tokens (list[str]): List of token strings.
        """
        for token in tokens:
            self.add_token(token)

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of token strings to their corresponding IDs.

        Args:
            tokens (list[str]): List of token strings.

        Returns:
            list[int]: Sequence of numerical token IDs.
        """
        return [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert a sequence of token IDs back into token strings.

        Args:
            ids (list[int]): Sequence of numerical token IDs.

        Returns:
            list[str]: Sequence of decoded token strings.
        """
        return [self.id2token.get(i, self.unk_token) for i in ids]

    def __len__(self) -> int:
        """Return the total number of unique tokens in the vocabulary.

        Returns:
            int: Size of the vocabulary.
        """
        return len(self.token2id)


# =====================================================================
# 2. ARCHITECTURE: CAUSAL MULTI-HEAD SELF-ATTENTION & DECODER BLOCK
# =====================================================================

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create an additive causal attention mask (-inf in the upper triangle).

    Ensures that position i can only attend to positions j <= i.

    Args:
        seq_len (int): Length of the sequence dimension.
        device (torch.device): Target device for the tensor allocation.

    Returns:
        torch.Tensor: Additive causal mask of shape (1, 1, seq_len, seq_len).
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Generate standard sinusoidal positional encodings.

    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Hidden dimension size.

    Returns:
        torch.Tensor: Fixed positional encoding matrix of shape (seq_len, d_model).
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    scale = -math.log(10000.0) / d_model
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * scale)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-Head Scaled Dot-Product Self-Attention with causal masking.

    Attributes:
        num_heads (int): Number of parallel attention heads.
        d_k (int): Feature dimension per individual attention head.
        w_q (nn.Linear): Query projection weight matrix.
        w_k (nn.Linear): Key projection weight matrix.
        w_v (nn.Linear): Value projection weight matrix.
        w_o (nn.Linear): Output projection weight matrix.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        """Initialize multi-head causal self-attention module.

        Args:
            d_model (int): Total model feature dimensionality. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.

        Raises:
            AssertionError: If d_model is not divisible by num_heads.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden dimensions across parallel attention heads.

        Args:
            x (torch.Tensor): Projected tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_len, d_k).
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        custom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute scaled dot-product attention with causal mask.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            custom_mask (torch.Tensor | None): Optional additive attention mask.
                If None, standard lower-triangular causal mask is constructed dynamically.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - out (torch.Tensor): Contextualized representation (batch_size, seq_len, d_model).
                - attn_weights (torch.Tensor): Attention probabilities (B, H, L, L).
        """
        batch_size, seq_len, _ = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if custom_mask is None:
            causal_mask = build_causal_mask(seq_len, x.device)
            scores = scores + causal_mask
        else:
            scores = scores + custom_mask

        attn_weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.w_o(out)
        return out, attn_weights


class CausalDecoderBlock(nn.Module):
    """Transformer decoder block with causal self-attention, residual streams, and FFN.

    Attributes:
        attention (MultiHeadCausalSelfAttention): Causal self-attention module.
        ln_1 (nn.LayerNorm): Layer normalization applied before self-attention.
        ln_2 (nn.LayerNorm): Layer normalization applied before feed-forward network.
        ffn (nn.Sequential): Position-wise feed-forward neural network.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None) -> None:
        """Initialize CausalDecoderBlock.

        Args:
            d_model (int): Hidden dimension size.
            num_heads (int): Number of parallel attention heads.
            d_ff (int | None): Intermediate dimension of feed-forward network. Defaults to 4 * d_model.
        """
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.ln_1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadCausalSelfAttention(d_model, num_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        custom_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute decoder block with pre-layer normalization and residual connections.

        Args:
            x (torch.Tensor): Input hidden states of shape (batch_size, seq_len, d_model).
            custom_mask (torch.Tensor | None): Optional additive causal mask.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - out (torch.Tensor): Updated hidden states (batch_size, seq_len, d_model).
                - attn_weights (torch.Tensor): Causal attention weights (B, H, L, L).
        """
        norm_x = self.ln_1(x)
        attn_out, attn_weights = self.attention(norm_x, custom_mask=custom_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, attn_weights


class MultiHeadCausalLM(nn.Module):
    """Decoder-only Causal Language Model based on multi-head self-attention.

    Attributes:
        embedding (nn.Embedding): Token embedding lookup table.
        decoder_block (CausalDecoderBlock): Transformer causal decoder block.
        ln_f (nn.LayerNorm): Final layer normalization before logit projection.
        decoder_head (nn.Linear): Linear projection layer to vocabulary logits.
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int) -> None:
        """Initialize MultiHeadCausalLM module.

        Args:
            vocab_size (int): Size of the token vocabulary.
            d_model (int): Dimensionality of embedding and attention features.
            num_heads (int): Number of parallel attention heads.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_block = CausalDecoderBlock(d_model, num_heads)
        self.ln_f = nn.LayerNorm(d_model)
        self.decoder_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute vocabulary logits, context vectors, and causal attention weights.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs with shape (B, L).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - logits (torch.Tensor): Unnormalized vocabulary scores with shape (B, L, V).
                - context_vectors (torch.Tensor): Hidden representations of shape (B, L, d_model).
                - attn_weights (torch.Tensor): Causal attention weights with shape (B, H, L, L).
        """
        batch_size, seq_len = input_ids.shape
        hidden = self.embedding(input_ids)
        pe = sinusoidal_positional_encoding(seq_len, hidden.size(-1)).to(hidden.device)
        hidden = hidden + pe.unsqueeze(0)

        context_vectors, attn_weights = self.decoder_block(hidden)
        norm_context = self.ln_f(context_vectors)
        logits = self.decoder_head(norm_context)
        return logits, norm_context, attn_weights


# =====================================================================
# 3. TRAINING SETUP (NEXT-TOKEN PREDICTION)
# =====================================================================

def prepare_causal_dataset(
    raw_tokens: list[str],
    vocab: Vocabulary,
    seq_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk tokens into uniform sequence lengths for next-token prediction.

    Prepares input-target pairs shifted by one position:
    Input:  [t_0, t_1, ..., t_{N-1}]
    Target: [t_1, t_2, ..., t_N]

    Args:
        raw_tokens (list[str]): List of raw token strings from text corpus.
        vocab (Vocabulary): Vocabulary used for encoding tokens into IDs.
        seq_len (int): Sequence length of the input window. Defaults to 16.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - inputs (torch.Tensor): Input token IDs of shape (N, seq_len).
            - targets (torch.Tensor): Target token IDs shifted by one of shape (N, seq_len).
    """
    token_ids = vocab.encode(raw_tokens)
    chunk_size = seq_len + 1
    num_seqs = len(token_ids) // chunk_size

    inputs, targets = [], []
    for i in range(num_seqs):
        chunk = token_ids[i * chunk_size : (i + 1) * chunk_size]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return inputs_tensor, targets_tensor


# =====================================================================
# 4. EVALUATION, INFERENCE & DIAGNOSTICS
# =====================================================================

def generate(
    model: MultiHeadCausalLM,
    vocab: Vocabulary,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate text autoregressively starting from a prompt with sampling controls.

    Args:
        model (MultiHeadCausalLM): Trained causal language model.
        vocab (Vocabulary): Vocabulary used for encoding and decoding tokens.
        prompt (str): Initial seed string for generation.
        max_new_tokens (int): Maximum number of tokens to generate. Defaults to 20.
        temperature (float): Softmax sampling temperature. Defaults to 1.0.
        top_k (int | None): Top-k filtering threshold. If None or <= 1, greedy decoding is used.
        top_p (float | None): Nucleus (top-p) cumulative probability threshold. Defaults to None.
        repetition_penalty (float): Penalty factor applied to previously generated tokens. Defaults to 1.2.

    Returns:
        str: Completed string including prompt and generated tokens.
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenize(prompt)
    if not tokens:
        tokens = [vocab.unk_token]

    input_ids = vocab.encode(tokens)
    generated_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated_ids], dtype=torch.long, device=device)
            logits, _, _ = model(x)
            next_token_logits = logits[0, -1, :].clone()

            # Apply repetition penalty to already emitted tokens
            if repetition_penalty != 1.0:
                for past_id in set(generated_ids):
                    if next_token_logits[past_id] > 0:
                        next_token_logits[past_id] /= repetition_penalty
                    else:
                        next_token_logits[past_id] *= repetition_penalty

            # Greedy or temperature-scaled sampling
            if temperature == 0.0 or (top_k == 1 and top_p is None):
                next_token_id = int(torch.argmax(next_token_logits).item())
            else:
                scaled_logits = next_token_logits / max(temperature, 1e-5)

                # Top-k filtering
                if top_k is not None and top_k > 1:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[-1]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_ids.append(next_token_id)

    decoded_tokens = vocab.decode(generated_ids)
    return " ".join(decoded_tokens)


def causal_witness_highlighter(
    model: MultiHeadCausalLM,
    vocab: Vocabulary,
    sentence: str,
    target_word: str,
) -> None:
    """Display causal attention distribution for a target token across all heads.

    Demonstrates that attention is strictly lower-triangular (tokens cannot attend
    to future tokens) and uses row-max normalized dynamic ASCII scaling.

    Args:
        model (MultiHeadCausalLM): Trained causal language model.
        vocab (Vocabulary): Vocabulary used for token encoding.
        sentence (str): Input context string.
        target_word (str): Target token whose attention distribution is inspected.
    """
    model.eval()
    tokens = tokenize(sentence)

    if target_word.lower() not in tokens:
        print(f"Target word '{target_word}' not found in sentence.")
        return

    device = next(model.parameters()).device
    input_ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, attn_matrix = model(input_ids)

    attn_matrix = attn_matrix.squeeze(0)  # (num_heads, seq_len, seq_len)
    target_idx = tokens.index(target_word.lower())

    print(f"\n--- Causal Witness Highlighter: Target = '{target_word}' (Pos {target_idx}) ---")
    print(f"Sentence: \"{' '.join(tokens)}\"")

    for head in range(attn_matrix.size(0)):
        head_weights = attn_matrix[head].tolist()
        row_weights = head_weights[target_idx]
        max_score = max(row_weights) if row_weights and max(row_weights) > 0 else 1.0

        print(f"\nHead {head} [Causal Self-Attention]")
        print(f"{'Context Token':<18} | {'Pos':<4} | {'Attention Score':<15} | Visual Weight")
        print("-" * 65)
        for idx, (token, score) in enumerate(zip(tokens, row_weights)):
            bar_len = int((score / max_score) * 25) if score > 0 else 0
            bar_repr = "█" * bar_len
            causal_tag = " (Future masked)" if idx > target_idx else ""
            print(f"{token:<18} | {idx:<4} | {score:.4f}{causal_tag:<16} | {bar_repr}")


def compute_corpus_contextual_baseline(
    model: MultiHeadCausalLM,
    vocab: Vocabulary,
    tokens: list[str],
    seq_len: int = 16,
    batch_size: int = 256,
    strategy: str = "last",
) -> torch.Tensor:
    """Compute the mean contextual representation across representative corpus chunks.

    Contextual vectors are extracted from the decoder representations matching
    the selected pooling strategy (last token or unpadded mean). Subtracting this
    baseline mitigates representation anisotropy (narrow cone phenomenon).

    Args:
        model (MultiHeadCausalLM): Trained causal language model.
        vocab (Vocabulary): Vocabulary used for encoding tokens into IDs.
        tokens (list[str]): Corpus token sequence.
        seq_len (int): Length of corpus chunks to process. Defaults to 16.
        batch_size (int): Batch size for parallel context extraction. Defaults to 256.
        strategy (str): Pooling strategy, either 'last' or 'mean'. Defaults to 'last'.

    Returns:
        torch.Tensor: The mean contextual baseline vector of shape (d_model,).
    """
    model.eval()
    device = next(model.parameters()).device
    token_ids = vocab.encode(tokens)
    num_seqs = len(token_ids) // seq_len

    if num_seqs == 0:
        return torch.zeros(model.embedding.embedding_dim, device=device)

    corpus_tensor = torch.tensor(
        token_ids[:num_seqs * seq_len], dtype=torch.long, device=device
    ).view(num_seqs, seq_len)

    all_pooled_vectors: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, num_seqs, batch_size):
            batch = corpus_tensor[i : i + batch_size]
            _, context_vectors, _ = model(batch)
            if strategy == "last":
                all_pooled_vectors.append(context_vectors[:, -1, :])
            else:
                all_pooled_vectors.append(context_vectors.mean(dim=1))

    return torch.cat(all_pooled_vectors, dim=0).mean(dim=0)


def get_phrase_embedding(
    model: MultiHeadCausalLM,
    vocab: Vocabulary,
    phrase: str,
    mean_baseline: torch.Tensor | None = None,
    strategy: str = "last",
) -> torch.Tensor:
    """Return a contextual embedding for a phrase using last-token or masked mean pooling.

    In causal models, 'last' pooling is the canonical representation because the
    final token has causally attended over all preceding tokens in the phrase.
    Optional mean-centering using a precomputed corpus baseline corrects for anisotropy.

    Args:
        model (MultiHeadCausalLM): Trained causal language model.
        vocab (Vocabulary): Vocabulary used for encoding tokens.
        phrase (str): Raw text phrase to embed.
        mean_baseline (torch.Tensor | None): Optional mean contextual baseline vector
            used for mean-centering anisotropy correction.
        strategy (str): Pooling strategy, either 'last' or 'mean'. Defaults to 'last'.

    Returns:
        torch.Tensor: Pooled representation vector with shape (d_model,).
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenize(phrase)

    token_ids = vocab.encode(tokens) if tokens else [vocab.token2id[vocab.unk_token]]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _, context_vectors, _ = model(input_ids)  # (1, seq_len, d_model)

    pad_id = vocab.token2id[vocab.pad_token]
    valid_mask = (input_ids != pad_id)

    if strategy == "last":
        # Extract the representation of the last unpadded token in the sequence
        seq_lengths = valid_mask.sum(dim=1)
        last_indices = (seq_lengths - 1).clamp(min=0)
        phrase_vector = context_vectors[0, last_indices[0]]
    else:
        # Masked mean pooling across unpadded tokens
        mask = valid_mask.unsqueeze(-1).float()
        mask_sum = mask.sum(dim=1).clamp(min=1.0)
        phrase_vector = (context_vectors * mask).sum(dim=1) / mask_sum
        phrase_vector = phrase_vector.squeeze(0)

    if mean_baseline is not None:
        phrase_vector = phrase_vector - mean_baseline

    return phrase_vector


def visualize_positional_encodings(
    d_model: int = 64,
    short_seq_len: int = 16,
    long_seq_len: int = 500,
    output_path: Path | None = None,
) -> None:
    """Visualize sinusoidal positional encodings over short and long horizons.

    Demonstrates "why low frequencies exist": over short sequences (N=16),
    higher-index embedding dimensions show little variation, completing full
    cycles only over extended sequence horizons (N=500).

    Args:
        d_model (int): Hidden dimension size. Defaults to 64.
        short_seq_len (int): Sequence length for short horizon. Defaults to 16.
        long_seq_len (int): Sequence length for long horizon. Defaults to 500.
        output_path (Path | None): File path to save generated heatmap figure.
    """
    pe_short = sinusoidal_positional_encoding(short_seq_len, d_model).numpy()
    pe_long = sinusoidal_positional_encoding(long_seq_len, d_model).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(pe_short, cmap="Blues", ax=axes[0], cbar=True)
    axes[0].set_title(f"Sinusoidal Positional Encoding (Short: N={short_seq_len}, d={d_model})")
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_ylabel("Sequence Position")

    sns.heatmap(pe_long, cmap="Blues", ax=axes[1], cbar=True)
    axes[1].set_title(f"Sinusoidal Positional Encoding (Long: N={long_seq_len}, d={d_model})")
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].set_ylabel("Sequence Position")

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved positional encoding visualization to: {output_path}")
    plt.close(fig)

    # Print channel variance diagnostic
    var_short = pe_short.var(axis=0)
    var_long = pe_long.var(axis=0)
    print("\n--- Positional Encoding Frequency Diagnostic ---")
    print(f"Dimension  0 (Highest freq) variance: Short={var_short[0]:.4f} | Long={var_long[0]:.4f}")
    print(f"Dimension {d_model - 1} (Lowest freq)  variance: Short={var_short[-1]:.4f} | Long={var_long[-1]:.4f}")


# =====================================================================
# 5. EXECUTION & EVALUATION PIPELINE
# =====================================================================

def run_evaluation(
    model: MultiHeadCausalLM,
    vocab: Vocabulary,
    tokens: list[str] | None = None,
    seq_len: int = 16,
) -> None:
    """Run generation, causal witness highlighting, and anisotropy phrase evaluation.

    Args:
        model (MultiHeadCausalLM): Trained causal language model.
        vocab (Vocabulary): Token vocabulary.
        tokens (list[str] | None): Optional tokenized corpus for computing contextual baseline.
        seq_len (int): Sequence length for corpus context extraction. Defaults to 16.
    """
    # 1. Autoregressive Generation Demonstration
    print("\n=======================================================")
    print("1. AUTOREGRESSIVE GENERATION (GREEDY, TOP-K & TOP-P)")
    print("=======================================================")
    test_prompts = [
        "the king said",
        "alice was beginning",
        "the queen of hearts",
        "down the rabbit",
    ]
    for prompt in test_prompts:
        greedy_gen = generate(
            model, vocab, prompt, max_new_tokens=12, temperature=0.0, repetition_penalty=1.2
        )
        sample_gen = generate(
            model, vocab, prompt, max_new_tokens=12, temperature=0.8, top_k=10, top_p=0.9, repetition_penalty=1.2
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"  Greedy (rep penalty=1.2): {greedy_gen}")
        print(f"  Sample (top-k=10, top-p=0.9): {sample_gen}")

    # 2. Causal Witness Highlighter
    print("\n=======================================================")
    print("2. CAUSAL WITNESS HIGHLIGHTER (LOWER TRIANGULAR MASK)")
    print("=======================================================")
    causal_sentence = "the queen of hearts she made some tarts"
    causal_witness_highlighter(model, vocab, causal_sentence, target_word="she")
    causal_witness_highlighter(model, vocab, causal_sentence, target_word="tarts")

    # 3. Anisotropy-Corrected Phrase Similarity: Last-Token vs Mean Pooling
    print("\n=======================================================")
    print("3. PHRASE EMBEDDINGS: LAST-TOKEN VS. MEAN-POOLING")
    print("=======================================================")
    phrase = "stole tarts"
    baseline_last: torch.Tensor | None = (
        compute_corpus_contextual_baseline(model, vocab, tokens, seq_len=seq_len, strategy="last")
        if tokens is not None
        else None
    )
    baseline_mean: torch.Tensor | None = (
        compute_corpus_contextual_baseline(model, vocab, tokens, seq_len=seq_len, strategy="mean")
        if tokens is not None
        else None
    )

    vec_raw_last = get_phrase_embedding(model, vocab, phrase, strategy="last")
    vec_cal_last = get_phrase_embedding(model, vocab, phrase, mean_baseline=baseline_last, strategy="last")

    vec_cal_mean = get_phrase_embedding(model, vocab, phrase, mean_baseline=baseline_mean, strategy="mean")

    Score = namedtuple("Score", ["calibrated_last", "phrase", "raw_last", "calibrated_mean"])
    scores: list[Score] = []
    other_phrases = [
        "King Hearts",
        "Queen Hearts",
        "Knave Hearts",
        "Alice",
        "White Rabbit",
        "Mad Hatter",
        "Cheshire Cat",
        "Caterpillar",
        "Dormouse",
        "March Hare",
        "Gryphon",
        "Mock Turtle",
        "Harry Potter",
    ]

    for other_phrase in other_phrases:
        vec_other_raw_last = get_phrase_embedding(model, vocab, other_phrase, strategy="last")
        vec_other_cal_last = get_phrase_embedding(
            model, vocab, other_phrase, mean_baseline=baseline_last, strategy="last"
        )
        vec_other_cal_mean = get_phrase_embedding(
            model, vocab, other_phrase, mean_baseline=baseline_mean, strategy="mean"
        )

        sim_raw_last = float(
            F.cosine_similarity(vec_raw_last.unsqueeze(0), vec_other_raw_last.unsqueeze(0)).item()  # pylint: disable=not-callable
        )
        sim_cal_last = float(
            F.cosine_similarity(vec_cal_last.unsqueeze(0), vec_other_cal_last.unsqueeze(0)).item()  # pylint: disable=not-callable
        )
        sim_cal_mean = float(
            F.cosine_similarity(vec_cal_mean.unsqueeze(0), vec_other_cal_mean.unsqueeze(0)).item()  # pylint: disable=not-callable
        )
        scores.append(
            Score(
                phrase=other_phrase,
                raw_last=sim_raw_last,
                calibrated_last=sim_cal_last,
                calibrated_mean=sim_cal_mean,
            )
        )

    scores.sort(key=lambda s: s.calibrated_last, reverse=True)
    print(f"\nTarget query: '{phrase}'")
    print(f"{'Comparison Phrase':<18} | {'Raw Last-Token':<14} | {'Calibrated Last':<16} | {'Calibrated Mean':<16}")
    print("-" * 72)
    for score in scores:
        print(
            f"{score.phrase:<18} | {score.raw_last:+.4f}         | {score.calibrated_last:+.4f}          | {score.calibrated_mean:+.4f}"
        )


def main(source_path: Path) -> None:
    """Execute training and evaluation pipeline for the toy causal self-attention model.

    Args:
        source_path (Path): Path to the source corpus text file.
    """
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    print(f"Using device: {device}")

    # 1. Load Data and Build Vocab
    raw_text = load_text(source_path)
    tokens = tokenize(raw_text)

    vocab = Vocabulary()
    vocab.build_vocab(tokens)
    print(f"Loaded text. Total tokens: {len(tokens)} | Vocab size: {len(vocab)}")

    # 2. Hyperparameters & Model Setup
    d_model = 64
    num_heads = 4
    seq_len = 16
    epochs = 250
    learning_rate = 0.003
    weight_decay = 0.01

    model = MultiHeadCausalLM(
        vocab_size=len(vocab),
        d_model=d_model,
        num_heads=num_heads,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pad_id = vocab.token2id[vocab.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 3. Prepare Causal Input-Target Pairs
    x_train, y_train = prepare_causal_dataset(tokens, vocab, seq_len=seq_len)
    x_train, y_train = x_train.to(device), y_train.to(device)
    print(f"Dataset prepared: {x_train.size(0)} sequence chunks of length {seq_len}")

    # 4. Optimization Loop
    print("\nStarting Training (Next-Token Prediction)...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _, _ = model(x_train)

        # Reshape logits to (N * L, V) and targets to (N * L)
        loss = criterion(logits.view(-1, len(vocab)), y_train.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | Causal LM Loss: {loss.item():.4f}")

    # 5. Evaluation & Diagnostics
    run_evaluation(model, vocab, tokens=tokens, seq_len=seq_len)

    # 6. Positional Encoding Visualization
    fig_path = Path(__file__).parent / "pe_visualization.png"
    visualize_positional_encodings(d_model=d_model, short_seq_len=seq_len, long_seq_len=500, output_path=fig_path)


if __name__ == "__main__":
    default_text_path = Path(__file__).parent / "../../data/book.txt"
    main(default_text_path)
