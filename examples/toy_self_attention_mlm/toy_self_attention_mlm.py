import math
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =====================================================================
# 1. DATA PREPARATION & TOKENIZATION
# =====================================================================

def load_text(text_path: Path):
    """Fetch the text."""
    return text_path.read_text(encoding='UTF-8')

def tokenize(text: str) -> list[str]:
    """Simple word-level tokenization preserving basic punctuation."""
    text = text.lower()
    text = text.replace('_', '')
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

class Vocabulary:
    def __init__(self, pad_token="[PAD]", mask_token="[MASK]", unk_token="[UNK]"):
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token

        self.special_tokens = [pad_token, mask_token, unk_token]
        self.token2id = {}
        self.id2token = {}

        for st in self.special_tokens:
            self.add_token(st)

    def add_token(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def build_vocab(self, tokens):
        for token in tokens:
            self.add_token(token)

    def encode(self, tokens):
        return [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]

    def decode(self, ids):
        return [self.id2token.get(i, self.unk_token) for i in ids]

    def __len__(self):
        return len(self.token2id)

# =====================================================================
# 2. ARCHITECTURE: MULTI-HEAD SELF-ATTENTION (FORCED HEAD ROLES) & MLM
# =====================================================================

# Head roles by index: 0 = local/positional, 1-2 = forced long-range/semantic, rest = free.
HEAD_ROLES = ["local/positional", "semantic/long-range", "semantic/long-range", "free"]

def build_head_masks(seq_len, num_heads, window_size, device):
    """Additive (-inf/0) masks per head enforcing the roles in HEAD_ROLES."""
    positions = torch.arange(seq_len, device=device)
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    local = distance <= window_size  # (seq_len, seq_len)

    NEG_INF = float("-inf")
    masks = []
    for head in range(num_heads):
        if head == 0:
            allowed = local
        elif head in (1, 2):
            allowed = ~local
        else:
            allowed = torch.ones_like(local)
        mask = torch.where(allowed, torch.zeros_like(distance, dtype=torch.float32), torch.full_like(distance, NEG_INF, dtype=torch.float32))
        # Guard against fully-masked rows (short sequences) which would make softmax produce NaN.
        fully_masked_rows = torch.isinf(mask).all(dim=-1)
        mask[fully_masked_rows] = 0.0
        masks.append(mask)
    return torch.stack(masks)  # (num_heads, seq_len, seq_len)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, X):
        batch_size, seq_len, _ = X.shape
        return X.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)

    def forward(self, X):
        # X shape: (batch_size, sequence_length, d_model)
        batch_size, seq_len, _ = X.shape
        Q = self._split_heads(self.W_q(X))
        K = self._split_heads(self.W_k(X))
        V = self._split_heads(self.W_v(X))

        # Scaled Dot-Product Attention: A = softmax(Q K^T / sqrt(d_k) + role_mask)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        role_mask = build_head_masks(seq_len, self.num_heads, self.window_size, X.device)
        A = torch.softmax(scores + role_mask, dim=-1)  # (B, H, L, L)

        # Context output: O = A V, heads concatenated back to d_model then projected
        O = torch.matmul(A, V)  # (B, H, L, d_k)
        O = O.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        O = self.W_o(O)
        return O, A

def sinusoidal_positional_encoding(seq_len, d_model):
    """Standard Transformer sin/cos positional encoding, shape (seq_len, d_model)."""
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class MultiHeadMLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, window_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, window_size)
        self.decoder_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        X = self.embedding(input_ids)
        # Computed per-call so it adapts to both fixed training windows and variable inference lengths.
        X = X + sinusoidal_positional_encoding(X.size(1), X.size(2)).to(X.device)
        O, A = self.attention(X)
        logits = self.decoder_head(O)
        return logits, A

# =====================================================================
# 3. TRAINING SETUP (MASKED LANGUAGE MODELING)
# =====================================================================

def prepare_mlm_dataset(raw_tokens, vocab, seq_len=16, mask_prob=0.15):
    """Chunk tokens into uniform sequence lengths and apply MLM masking."""
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

def get_phrase_embedding(model: MultiHeadMLM, vocab: Vocabulary, phrase: str):
    """Return a mean-pooled contextual embedding for ``phrase``."""
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenize(phrase)
    input_ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)

    with torch.no_grad():
        # Step 1: Input Embeddings + Positional Encodings
        X = model.embedding(input_ids)
        X = X + sinusoidal_positional_encoding(X.size(1), X.size(2)).to(device)

        # Step 2: Contextualized output representations from Attention
        context_vectors, _ = model.attention(X) # Shape: (1, seq_len, d_model)

    # Pool sequence vectors across time (Mean Pooling) to get a single vector for the phrase
    phrase_vector = context_vectors.squeeze(0).mean(dim=0)
    return phrase_vector

def witness_highlighter(model: MultiHeadMLM, vocab: Vocabulary, sentence, target_word):
    """
    Passes a sentence through the trained model, isolates the attention row
    for target_word, and prints the context weight distribution.
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
        attn_weights = attn_weights[target_idx]
        print(f"\nHead {head} [{role}]")
        print(f"{'Context Token':<18} | {'Attention Score':<15} | Visual Weight")
        print("-" * 55)
        for token, score in zip(tokens, attn_weights):
            bar = "█" * int(score * 30)
            print(f"{token:<18} | {score:.4f}          | {bar}")


# =====================================================================
# 5. EXECUTION PIPELINE
# =====================================================================

def main(text_path: Path):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    print(f"Using device: {device}")

    # 1. Load Data and Build Vocab
    raw_text = load_text(text_path)
    tokens = tokenize(raw_text)

    vocab = Vocabulary()
    vocab.build_vocab(tokens)
    print(f"Loaded text. Total tokens: {len(tokens)} | Vocab size: {len(vocab)}")

    # 2. Hyperparameters & Model Setup
    D_MODEL = 64
    NUM_HEADS = 4
    WINDOW_SIZE = 2
    SEQ_LEN = 16
    EPOCHS = 200
    LR = 0.005

    model = MultiHeadMLM(vocab_size=len(vocab), d_model=D_MODEL, num_heads=NUM_HEADS, window_size=WINDOW_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 3. Prepare MLM Inputs
    X_train, Y_train = prepare_mlm_dataset(tokens, vocab, seq_len=SEQ_LEN)
    X_train, Y_train = X_train.to(device), Y_train.to(device)

    # 4. Optimization Loop
    print("\nStarting Training...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits, _ = model(X_train)

        # Reshape logits to (N * L, V) and labels to (N * L) for CrossEntropyLoss
        loss = criterion(logits.view(-1, len(vocab)), Y_train.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | MLM Loss: {loss.item():.4f}")

    # 5. Evaluate with the Witness Highlighter
    sample_sentence = "The king said gravely consider your verdict"
    witness_highlighter(model, vocab, sample_sentence, target_word="verdict")
    witness_highlighter(model, vocab, sample_sentence, target_word="king")
    sample_sentence = "the knave of hearts he stole those tarts"
    witness_highlighter(model, vocab, sample_sentence, target_word="he")
    witness_highlighter(model, vocab, sample_sentence, target_word="stole")
    sample_sentence = "the judge said calmly consider your sentence"
    witness_highlighter(model, vocab, sample_sentence, target_word="sentence")

    # 6. Check embeddings
    print("\nChecking phrase embeddings...")
    phrase = "stole tarts"
    vec = get_phrase_embedding(model, vocab, phrase)
    other_phrases = [
        # "The King of Hearts",
        # "The Queen of Hearts",
        # "The Knave of Hearts",
        # "Alice",
        # "The White Rabbit",
        # "The Mad Hatter",
        # "The Cheshire Cat",
        # "The Caterpillar",
        # "The Dormouse",
        # "The March Hare",
        # "the Gryphon",
        # "the Mock Turtle",
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
        "Wakawaka",
    ]
    for other_phrase in other_phrases:
        vec_other = get_phrase_embedding(model, vocab, other_phrase)
        # Compute similarity (-1.0 to 1.0)
        similarity = F.cosine_similarity(vec.unsqueeze(0), vec_other.unsqueeze(0)).item()
        print(f"Similarity between '{phrase}' and '{other_phrase}': {similarity:.4f}")


if __name__ == "__main__":
    text_path = Path(__file__).parent / '../../data/book.txt'
    main(text_path)
