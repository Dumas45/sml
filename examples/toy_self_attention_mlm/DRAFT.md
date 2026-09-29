# Draft: Mechanics of Self-Attention & The Witness Highlighter

### Intent

I want to understand the Mechanics of Self-Attention in machine learning: how it works, why it works, and what semantic relationships it learns.

To explore this, I will implement and train a toy model in PyTorch using the text of the book *Alice in Wonderland***.

### Technical Scope

* **Architecture:** Explicit multi-head self-attention (`MultiHeadSelfAttention`) with sinusoidal positional encodings and custom head-role attention masking (`build_head_masks`).

* **Head Roles:** Enforced roles per head index (e.g., local/positional, forced semantic/long-range, and free) using additive distance-based masks.

* **Exclusions:** Pre-trained language models or standard token classification heads.

---

# Proposition: The Witness Highlighter (Contextual Retrieval)

Instead of a classification task, the model is evaluated using an interpretable, unsupervised retrieval task: **The Witness Highlighter**.

The book features distinct entities (Alice, King, Queen, Knave, Cook, Dormouse) participating in structured narrative events. The objective is to evaluate whether multi-head attention learns syntactic or semantic bindings directly from raw text.

### The Mechanism

1. Input a tokenized sentence from the book (e.g., *"the king said gravely consider your verdict"*).


2. Compute the multi-head attention tensor:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{role\_mask}\right) \in \mathbb{R}^{H \times N \times N}$$

3. For a target token (e.g., `"verdict"`), extract its corresponding attention vectors across all heads $A_{h, \text{target}, :}$.

4. High attention scores assigned to contextually relevant tokens (e.g., `"gravely"`, `"said"`, or `"consider"`) across structured heads demonstrate successful contextual binding without explicit labeled supervision.

---

# Core Self-Attention Pipeline

For an input sequence $X \in \mathbb{R}^{B \times N \times d_{\text{model}}}$ of sequence length $N$ and $H$ heads ($d_k = d_{\text{model}} / H$):

1. **Embedding Lookup & Positional Encoding:** Map input token IDs to continuous representations and add sinusoidal positional encodings:

$$X_{\text{pos}} = \text{Embedding}(X) + \text{SinusoidalPE}(N, d_{\text{model}})$$

2. **Linear Projections & Multi-Head Split:** Compute Query ($Q$), Key ($K$), and Value ($V$) tensors reshaped to $(B, H, N, d_k)$:

$$Q = \text{Split}(X_{\text{pos}} W_q), \quad K = \text{Split}(X_{\text{pos}} W_k), \quad V = \text{Split}(X_{\text{pos}} W_v)$$

3. **Scaled Dot-Product Attention with Head Role Masks:**

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{role\_mask}\right)$$

$$O = \text{Concat}(AV) W_o$$

---

# Training Strategy

Because self-attention is an intermediate aggregation step rather than a loss function, the projection weights ($W_q, W_k, W_v, W_o$) and embeddings are trained using an unsupervised Masked Language Modeling (MLM) objective.

### Selected Approach: Masked Language Modeling (MLM)

* **Procedure:** Tokenize raw text, chunk into sequences of length $N = 16$, and randomly mask 15% of tokens with a `[MASK]` token.

* **Architecture:** Pass sequences through `MultiHeadMLM` (`Embedding` $\to$ `MultiHeadSelfAttention` $\to$ `decoder_head` $W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times \vert{}V\vert{}}$).

* **Loss Function:** Compute Cross-Entropy Loss exclusively on the masked positions (setting target labels to `-100` for non-masked positions):

$$\mathcal{L} = -\sum_{i \in \text{masked}} \log P(x_i \mid X_{\backslash i})$$

---

# Key Changes Summary

1. **Single-Head to Multi-Head Attention:** Updated specifications from single-head self-attention to multi-head self-attention with 4 heads (`num_heads=4`, `d_model=64`, `d_k=16`).

2. **Forced Head Role Masking:** Documented `build_head_masks`, which applies distance-based additive masks to enforce specific roles (local vs. long-range) per attention head.

3. **Positional Encodings:** Explicitly added sinusoidal positional encodings ($X + \text{PE}$) to the input embedding pipeline.

4. **Data Preprocessing & Chunking:** Clarified word-level tokenization preserving punctuation and fixed-window chunking ($L=16$) for MLM dataset preparation.
