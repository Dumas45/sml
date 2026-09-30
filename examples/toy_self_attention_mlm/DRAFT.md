# Updated Draft: Mechanics of Self-Attention & The Witness Highlighter

### Intent

This document aims to explore the Mechanics of Self-Attention in machine learning: how it works, why it works, and what semantic relationships it learns.

To explore this, a toy model will be implemented and trained in PyTorch using the text of the book *Alice in Wonderland*.

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

During MLM training, positional encodings are zeroed out at `[MASK]` token positions to prevent position leakage.

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

* **Position Leakage Prevention:** In standard self-attention, adding unmasked positional encodings to `[MASK]` positions allows the network to exploit absolute position indices as a shortcut for predicting the masked word without attending to surrounding context. In `MultiHeadMLM.forward`, positional encodings are explicitly masked out (zeroed via `pe.masked_fill(is_masked, 0.0)`) for all `[MASK]` positions, compelling the attention layers to rely strictly on contextual cues from neighboring unmasked tokens.

* **Loss Function:** Compute Cross-Entropy Loss exclusively on the masked positions (setting target labels to `-100` for non-masked positions):

$$\mathcal{L} = -\sum_{i \in \text{masked}} \log P(x_i \mid X_{\backslash i})$$

---

# Representation Evaluation & Visualization

### Dynamic Scaling in Witness Highlighter

In `witness_highlighter`, rather than using a static length multiplier, the visual ASCII bar length scales dynamically relative to the maximum attention score in the target row:

$$\text{bar\_length} = \text{int}\left(\frac{A_{h, \text{target}, j}}{\max_k A_{h, \text{target}, k}} \times 25\right)$$

This relative normalization preserves visibility for smaller, nuanced attention weights across both peaked and diffuse attention heads.

### Padding-Aware Mean Pooling

In `get_phrase_embedding`, phrase-level embeddings are computed by pooling token representations from the attention layer. An explicit attention/padding mask $m \in \{0, 1\}^{B \times N \times 1}$ is applied so that padding tokens `[PAD]` or variable sequence lengths do not distort phrase averages or produce division-by-zero errors:

$$\mathbf{v}_{\text{phrase}} = \frac{\sum_{i=1}^N m_i \cdot \mathbf{c}_i}{\max\left(1, \sum_{i=1}^N m_i\right)}$$

### Anisotropy Calibration (Mean-Centering)

Contextual representations generated by un-normalized toy transformer models on small corpora often suffer from **representation anisotropy** (the "narrow cone" phenomenon), where token vectors cluster in a narrow positive sub-cone of the embedding space. This causes high baseline cosine similarity even between unrelated phrases or nonsense control tokens (e.g., `"Wakawaka"` showing baseline similarity near $\sim 0.46$).

Because `phrase_vector` is produced after positional encoding, multi-head attention ($W_q, W_k, W_v$), and output projection ($W_o$), subtracting a static input embedding baseline operates across mismatched learned bases and fails to center the contextual vector distribution.

To calibrate semantic retrieval, the model computes the mean contextual baseline vector across representative corpus chunks passed through the exact same attention and pooling pipeline:

$$\bar{\mathbf{c}}_{\text{corpus}} = \frac{1}{M} \sum_{j=1}^M \mathbf{c}_j$$

where $\mathbf{c}_j$ is the pooled contextual representation of corpus chunk $j$. Subtracting this contextual baseline centers the contextual representation space:

$$\mathbf{v}'_{\text{phrase}} = \mathbf{v}_{\text{phrase}} - \bar{\mathbf{c}}_{\text{corpus}}$$

This mean-centering adjustment exposes cleaner relative semantic clustering across character entities and narrative roles (e.g., unrelated control phrases drop from positive similarities near $+0.46$ down to $+0.20$, while true antagonists like `"Knave Hearts"` maintain high relative similarity of $+0.57$).

---

# Pedagogical Takeaways for Lecture

The implementation provides three direct discussion points for teaching self-attention:

1. **"Why low frequencies exist":** Sinusoidal positional encodings trade off local distance discrimination (high frequencies in lower embedding dimensions) with global sequence stability (low frequencies in higher dimensions).
2. **"Why toy models overfit & anisotropy":** Training on a single book chapter demonstrates how small corpora and un-normalized representations lead to vector anisotropy, and how mean-centering recovers interpretable cosine distances.
3. **"Inductive bias via masks":** Explicit distance masks (`build_head_masks`) demonstrate how architectural inductive bias can force head specialization (local syntactic binding vs. long-range semantic binding) even when training data is minimal.

---

# Implemented Improvements & Key Changes Summary

1. **Multi-Head Attention Architecture:** Upgraded from single-head to 4-head self-attention (`num_heads=4`, `d_model=64`, `d_k=16`) with explicit head projection and concatenation.
2. **Forced Head Role Masking:** Documented `build_head_masks`, applying additive distance masks to enforce local syntactic tracking (head 0), long-range semantic tracking (heads 1–2), and unrestricted attention (head 3).
3. **Positional Encoding Leakage Prevention:** Zeroed out positional encodings at `[MASK]` positions in `MultiHeadMLM` to prevent positional shortcut learning and enforce genuine contextual reconstruction.
4. **Padding-Aware Phrase Pooling:** Added explicit sequence length and padding masks in `get_phrase_embedding` to avoid pooling skews and edge-case division-by-zero.
5. **Anisotropy Mitigation:** Introduced `compute_corpus_contextual_baseline` to compute the baseline across representative corpus contexts using the exact attention pipeline, and mean-centered phrase vectors to correct for contextual space anisotropy and calibrate cosine similarity evaluations.
6. **Dynamic Attention Visualization:** Replaced static ASCII bar scaling in `witness_highlighter` with row-max relative scaling for improved terminal log legibility.
