"""Toy Multi-Head Self-Attention trained with Masked Language Modeling (MLM).

This educational module implements an explicit multi-head self-attention network
trained on raw text from Alice in Wonderland to demonstrate:
1. "Why low frequencies exist": How sinusoidal positional encodings trade off
   local token distance discrimination with global sequence stability.
2. "Why toy models overfit & anisotropy": How training on minimal text causes
   contextual embeddings to occupy a narrow cone (anisotropy), yielding high
   baseline cosine similarity even for nonsense tokens, and how mean-centering
   calibrates semantic retrieval.
3. "Inductive bias via masks": How distance-based additive role masks enforce
   specialized head functions (local/positional vs. long-range/semantic).
"""

from collections import namedtuple
import math
from pathlib import Path
import re

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
    return text_path.read_text(encoding='UTF-8')


def tokenize(text: str) -> list[str]:
    """Tokenize input text at word level while preserving basic punctuation.

    Args:
        text (str): Raw string to tokenize.

    Returns:
        list[str]: Extracted list of lowercase tokens and punctuation marks.
    """
    text = text.lower()
    text = text.replace('_', '')
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


class Vocabulary:
    """Vocabulary mapping between unique tokens and numerical indices.

    Attributes:
        pad_token (str): Special token representing padding.
        mask_token (str): Special token representing masked inputs.
        unk_token (str): Special token representing out-of-vocabulary words.
        special_tokens (list[str]): List of all registered special tokens.
        token2id (dict[str, int]): Mapping from token strings to integer IDs.
        id2token (dict[int, str]): Mapping from integer IDs to token strings.
    """

    def __init__(
        self,
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        unk_token: str = "[UNK]",
    ) -> None:
        """Initialize Vocabulary with special tokens.

        Args:
            pad_token (str): Symbol for padding token. Defaults to '[PAD]'.
            mask_token (str): Symbol for mask token. Defaults to '[MASK]'.
            unk_token (str): Symbol for unknown token. Defaults to '[UNK]'.
        """
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token

        self.special_tokens = [pad_token, mask_token, unk_token]
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
# 2. ARCHITECTURE: MULTI-HEAD SELF-ATTENTION (FORCED HEAD ROLES) & MLM
# =====================================================================

# Head roles by index: 0 = local/positional, 1-2 = forced long-range/semantic, rest = free.
HEAD_ROLES = ["local/positional", "semantic/long-range", "semantic/long-range", "free"]


def build_head_masks(
    seq_len: int,
    num_heads: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create additive attention masks (-inf / 0.0) enforcing head roles.

    Enforces local inductive bias on head 0 (within window_size), long-range
    bias on heads 1-2 (strictly outside window_size), and unrestricted attention
    on remaining heads.

    Args:
        seq_len (int): Sequence length of the attention matrix.
        num_heads (int): Total number of attention heads.
        window_size (int): Distance threshold for local vs. long-range interactions.
        device (torch.device): Device on which tensors are allocated.

    Returns:
        torch.Tensor: Additive attention masks of shape (num_heads, seq_len, seq_len).
    """
    positions = torch.arange(seq_len, device=device)
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    local = distance <= window_size  # (seq_len, seq_len)

    neg_inf = float("-inf")
    masks = []
    for head in range(num_heads):
        if head == 0:
            allowed = local
        elif head in (1, 2):
            allowed = ~local
        else:
            allowed = torch.ones_like(local)
        mask = torch.where(
            allowed,
            torch.zeros_like(distance, dtype=torch.float32),
            torch.full_like(distance, neg_inf, dtype=torch.float32),
        )
        # Guard against fully-masked rows (e.g. short sequences) which make softmax produce NaN.
        fully_masked_rows = torch.isinf(mask).all(dim=-1)
        mask[fully_masked_rows] = 0.0
        masks.append(mask)
    return torch.stack(masks)  # (num_heads, seq_len, seq_len)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Scaled Dot-Product Self-Attention with role masking.

    Attributes:
        num_heads (int): Number of parallel attention heads.
        d_k (int): Feature dimension per individual attention head.
        window_size (int): Local window distance threshold for role masks.
        w_q (nn.Linear): Query projection weight matrix.
        w_k (nn.Linear): Key projection weight matrix.
        w_v (nn.Linear): Value projection weight matrix.
        w_o (nn.Linear): Output projection weight matrix.
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int) -> None:
        """Initialize multi-head self-attention module.

        Args:
            d_model (int): Total model feature dimensionality. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            window_size (int): Distance threshold used to construct role masks.

        Raises:
            AssertionError: If d_model is not divisible by num_heads.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden dimensions across parallel attention heads.

        Args:
            x (torch.Tensor): Projected tensor with shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Reshaped tensor with shape (batch_size, num_heads, seq_len, d_k).
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute scaled dot-product attention with head role masks.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - out (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
                - attn_weights (torch.Tensor): Attention weights (B, H, L, L).
        """
        batch_size, seq_len, _ = x.shape
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        # Scaled Dot-Product Attention: A = softmax(Q K^T / sqrt(d_k) + role_mask)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        role_mask = build_head_masks(seq_len, self.num_heads, self.window_size, x.device)
        attn_weights = torch.softmax(scores + role_mask, dim=-1)  # (B, H, L, L)

        # Context output: O = A V, heads concatenated back to d_model then projected
        out = torch.matmul(attn_weights, v)  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.w_o(out)
        return out, attn_weights


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

class MultiHeadMLM(nn.Module):
    """Masked Language Model based on explicit multi-head self-attention.

    Attributes:
        embedding (nn.Embedding): Token embedding lookup table.
        attention (MultiHeadSelfAttention): Multi-head self-attention module.
        decoder_head (nn.Linear): Linear projection layer to vocabulary logits.
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, window_size: int) -> None:
        """Initialize the MultiHeadMLM module.

        Args:
            vocab_size (int): Size of the token vocabulary.
            d_model (int): Dimensionality of embedding and attention features.
            num_heads (int): Number of parallel attention heads.
            window_size (int): Local window distance threshold for role masks.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, window_size)
        self.decoder_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask_token_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and multi-head attention weights for input tokens.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs with shape (B, L).
            mask_token_id (int | None): ID of [MASK] token. When provided,
                positional encodings are zeroed out at masked positions to prevent
                the model from exploiting absolute positions as an MLM shortcut.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - logits (torch.Tensor): Unnormalized vocabulary scores with shape (B, L, V).
                - attn_weights (torch.Tensor): Attention weights with shape (B, H, L, L).
        """
        hidden = self.embedding(input_ids)
        pe = sinusoidal_positional_encoding(hidden.size(1), hidden.size(2)).to(hidden.device)

        if mask_token_id is not None:
            # Mask out positional features where tokens are [MASK] to prevent position leakage.
            is_masked = (input_ids == mask_token_id).unsqueeze(-1)  # (B, L, 1)
            pe = pe.unsqueeze(0).expand_as(hidden)
            pe = pe.masked_fill(is_masked, 0.0)
            hidden = hidden + pe
        else:
            hidden = hidden + pe

        out, attn_weights = self.attention(hidden)
        logits = self.decoder_head(out)
        return logits, attn_weights

# =====================================================================
# 3. TRAINING SETUP (MASKED LANGUAGE MODELING)
# =====================================================================

def prepare_mlm_dataset(
    raw_tokens: list[str],
    vocab: Vocabulary,
    seq_len: int = 16,
    mask_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk tokens into uniform sequence lengths and apply MLM masking.

    Args:
        raw_tokens (list[str]): List of raw token strings from text corpus.
        vocab (Vocabulary): Vocabulary used for encoding tokens into IDs.
        seq_len (int): Sequence length of each chunk. Defaults to 16.
        mask_prob (float): Probability of masking each token. Defaults to 0.15.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - inputs (torch.Tensor): Token IDs of shape (N, seq_len).
            - labels (torch.Tensor): Target labels (-100 for unmasked) of shape (N, seq_len).
    """
    token_ids = vocab.encode(raw_tokens)
    num_seqs = len(token_ids) // seq_len

    inputs, labels = [], []
    mask_id = vocab.token2id[vocab.mask_token]

    for i in range(num_seqs):
        seq = torch.tensor(token_ids[i * seq_len : (i + 1) * seq_len], dtype=torch.long)
        label = seq.clone()

        # Randomly select 15% of indices to mask
        probability_matrix = torch.full(seq.shape, mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set target label to -100 for non-masked tokens (ignored by CrossEntropyLoss)
        label[~masked_indices] = -100

        # Replace selected tokens with [MASK]
        seq[masked_indices] = mask_id

        inputs.append(seq)
        labels.append(label)

    return torch.stack(inputs), torch.stack(labels)

# =====================================================================
# 4. WITNESS HIGHLIGHTER (EVALUATION & INFERENCE)
# =====================================================================

def compute_corpus_contextual_baseline(
    model: MultiHeadMLM,
    vocab: Vocabulary,
    tokens: list[str],
    seq_len: int = 16,
    batch_size: int = 256,
) -> torch.Tensor:
    """Compute the mean contextual representation across representative corpus chunks.

    Contextual vectors are extracted from the same embedding, positional encoding,
    and self-attention pooling pipeline as `get_phrase_embedding`. Subtracting this
    vector mean-centers the contextual representation distribution to mitigate
    representation anisotropy (narrow cone phenomenon).

    Args:
        model (MultiHeadMLM): Trained multi-head masked language model.
        vocab (Vocabulary): Vocabulary used for encoding tokens into IDs.
        tokens (list[str]): Corpus token sequence.
        seq_len (int): Length of corpus chunks to process. Defaults to 16.
        batch_size (int): Batch size for parallel context extraction. Defaults to 256.

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
    pe = sinusoidal_positional_encoding(seq_len, model.embedding.embedding_dim).to(device)

    with torch.no_grad():
        for i in range(0, num_seqs, batch_size):
            batch = corpus_tensor[i : i + batch_size]
            hidden = model.embedding(batch) + pe
            context_vectors, _ = model.attention(hidden)
            all_pooled_vectors.append(context_vectors.mean(dim=1))

    return torch.cat(all_pooled_vectors, dim=0).mean(dim=0)


def get_phrase_embedding(
    model: MultiHeadMLM,
    vocab: Vocabulary,
    phrase: str,
    mean_baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a mean-pooled contextual embedding for a phrase.

    Applies an attention/padding mask during pooling to avoid skew from padding
    or zero-length sequences, and optionally mean-centers the vector using a
    contextual baseline to correct for representation anisotropy.

    Args:
        model (MultiHeadMLM): The trained language model.
        vocab (Vocabulary): Vocabulary used for encoding tokens.
        phrase (str): Raw text phrase to embed.
        mean_baseline (torch.Tensor | None): Optional mean contextual baseline vector
            from the attention pipeline used for mean-centering anisotropy correction.

    Returns:
        torch.Tensor: Pooled representation vector with shape (d_model,).
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenize(phrase)

    # Encode tokens; if phrase is empty, fall back to unk token
    token_ids = vocab.encode(tokens) if tokens else [vocab.token2id[vocab.unk_token]]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        # Step 1: Input Embeddings + Positional Encodings
        hidden = model.embedding(input_ids)
        hidden = hidden + sinusoidal_positional_encoding(hidden.size(1), hidden.size(2)).to(device)

        # Step 2: Contextualized output representations from Attention
        context_vectors, _ = model.attention(hidden)  # Shape: (1, seq_len, d_model)

    # Step 3: Explicit attention/padding mask for pooling
    pad_id = vocab.token2id[vocab.pad_token]
    mask = (input_ids != pad_id).unsqueeze(-1).float()  # (1, seq_len, 1)
    mask_sum = mask.sum(dim=1).clamp(min=1.0)  # (1, 1)
    phrase_vector = (context_vectors * mask).sum(dim=1) / mask_sum
    phrase_vector = phrase_vector.squeeze(0)

    # Step 4: Optional anisotropy adjustment (mean-centering)
    if mean_baseline is not None:
        phrase_vector = phrase_vector - mean_baseline

    return phrase_vector

def witness_highlighter(
    model: MultiHeadMLM,
    vocab: Vocabulary,
    sentence: str,
    target_word: str,
) -> None:
    """Pass a sentence through the model and print the attention distribution for a target token.

    Dynamic scaling is applied to the ASCII bar visualization based on the maximum attention
    weight in each row so that smaller, nuanced weights remain visually readable.

    Args:
        model (MultiHeadMLM): The trained language model.
        vocab (Vocabulary): Vocabulary used for encoding tokens.
        sentence (str): Input sentence to evaluate.
        target_word (str): Target token whose attention distribution is displayed.
    """
    model.eval()
    tokens = tokenize(sentence)

    if target_word.lower() not in tokens:
        print(f"Target word '{target_word}' not found in sentence.")
        return

    device = next(model.parameters()).device
    input_ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)

    with torch.no_grad():
        _, attn_matrix = model(input_ids)

    attn_matrix = attn_matrix.squeeze(0)  # (num_heads, seq_len, seq_len)
    target_idx = tokens.index(target_word.lower())

    print(f"\n--- Witness Highlighter: Target = '{target_word}' ---")
    print(f"Sentence: \"{' '.join(tokens)}\"")

    for head, attn_weights in enumerate(attn_matrix.tolist()):
        role = HEAD_ROLES[head] if head < len(HEAD_ROLES) else "free"
        row_weights = attn_weights[target_idx]
        max_score = max(row_weights) if row_weights and max(row_weights) > 0 else 1.0

        print(f"\nHead {head} [{role}]")
        print(f"{'Context Token':<18} | {'Attention Score':<15} | Visual Weight")
        print("-" * 55)
        for token, score in zip(tokens, row_weights):
            bar_len = int((score / max_score) * 25) if score > 0 else 0
            bar_repr = "█" * bar_len
            print(f"{token:<18} | {score:.4f}          | {bar_repr}")


# =====================================================================
# 5. EXECUTION PIPELINE
# =====================================================================

def run_evaluation(
    model: MultiHeadMLM,
    vocab: Vocabulary,
    tokens: list[str] | None = None,
    seq_len: int = 16,
) -> None:
    """Run witness highlighter and anisotropy-corrected phrase similarity evaluation.

    Args:
        model (MultiHeadMLM): Trained multi-head masked language model.
        vocab (Vocabulary): Token vocabulary.
        tokens (list[str] | None): Optional tokenized corpus for computing the
            contextual mean baseline across attention representations.
        seq_len (int): Sequence length for corpus context extraction. Defaults to 16.
    """
    # 1. Evaluate with the Witness Highlighter
    sample_sentence = "The king said gravely consider your verdict"
    witness_highlighter(model, vocab, sample_sentence, target_word="verdict")
    witness_highlighter(model, vocab, sample_sentence, target_word="king")
    sample_sentence = "the knave of hearts he stole those tarts"
    witness_highlighter(model, vocab, sample_sentence, target_word="he")
    witness_highlighter(model, vocab, sample_sentence, target_word="stole")
    sample_sentence = "the judge said calmly consider your sentence"
    witness_highlighter(model, vocab, sample_sentence, target_word="sentence")

    # 2. Check embeddings: Raw vs. Anisotropy-Corrected (Mean-Centered)
    print("\nChecking phrase embeddings (raw vs. anisotropy-corrected)...")
    phrase = "stole tarts"
    if tokens is not None:
        mean_baseline: torch.Tensor | None = compute_corpus_contextual_baseline(
            model, vocab, tokens, seq_len=seq_len
        )
    else:
        mean_baseline = None

    vec_raw = get_phrase_embedding(model, vocab, phrase)
    vec_calibrated = get_phrase_embedding(model, vocab, phrase, mean_baseline=mean_baseline)

    Score = namedtuple("Score", ["calibrated", "phrase", "raw"])
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
    print(f"\nTarget query: '{phrase}'")
    print(f"{'Comparison Phrase':<18} | {'Raw CosSim':<12} | {'Calibrated CosSim':<17}")
    print("-" * 55)
    for other_phrase in other_phrases:
        vec_other_raw = get_phrase_embedding(model, vocab, other_phrase)
        vec_other_calibrated = get_phrase_embedding(
            model, vocab, other_phrase, mean_baseline=mean_baseline
        )

        sim_raw = float(
            F.cosine_similarity(vec_raw.unsqueeze(0), vec_other_raw.unsqueeze(0)).item()  # pylint: disable=not-callable
        )
        sim_cal = float(
            F.cosine_similarity(  # pylint: disable=not-callable
                vec_calibrated.unsqueeze(0), vec_other_calibrated.unsqueeze(0)
            ).item()
        )
        scores.append(Score(phrase=other_phrase, raw=sim_raw, calibrated=sim_cal))
    scores.sort(reverse=True)
    for score in scores:
        print(f"{score.phrase:<18} | {score.raw:+.4f}      | {score.calibrated:+.4f}")


def main(source_path: Path) -> None:
    """Execute training and evaluation pipeline for the toy MLM self-attention model.

    Args:
        source_path (Path): Path to the source corpus text file.
    """
    # Set random seed for reproducibility
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
    window_size = 2
    seq_len = 16
    epochs = 200
    learning_rate = 0.005

    model = MultiHeadMLM(
        vocab_size=len(vocab),
        d_model=d_model,
        num_heads=num_heads,
        window_size=window_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 3. Prepare MLM Inputs
    x_train, y_train = prepare_mlm_dataset(tokens, vocab, seq_len=seq_len)
    x_train, y_train = x_train.to(device), y_train.to(device)
    mask_token_id = vocab.token2id[vocab.mask_token]

    # 4. Optimization Loop
    print("\nStarting Training...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(x_train, mask_token_id=mask_token_id)

        # Reshape logits to (N * L, V) and labels to (N * L) for CrossEntropyLoss
        loss = criterion(logits.view(-1, len(vocab)), y_train.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | MLM Loss: {loss.item():.4f}")

    run_evaluation(model, vocab, tokens=tokens, seq_len=seq_len)


if __name__ == "__main__":
    default_text_path = Path(__file__).parent / '../../data/book.txt'
    main(default_text_path)
