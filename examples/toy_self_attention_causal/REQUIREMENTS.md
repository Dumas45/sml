## Requirements to create a new Causal Language Model

Here are the structured requirements to create a new Causal Language Model (GPT-style) (`examples/toy_self_attention_causal/toy_self_attention_causal.py`) implementation on the *Alice in Wonderland* text while leaving the existing MLM code (`examples/toy_self_attention_mlm/toy_self_attention_mlm.py`) intact.

---

### 1. File Structure & Scope

* **New Script Creation:** Create a new Python script named `examples/toy_self_attention_causal/toy_self_attention_causal.py` that operates independently without modifying or overwriting any logic in `examples/toy_self_attention_mlm/toy_self_attention_mlm.py`.


---

### 2. Data Preparation & Vocabulary Requirements

* **Text Tokenization:** Retain the tokenization pipeline (lowercasing, punctuation preservation, and word-level extraction).


* **Vocabulary Adaptation:**
* Remove `[MASK]` from the special tokens, as masked tokens are not used in causal language modeling.


* Keep `[PAD]` and `[UNK]` for handling padding and out-of-vocabulary tokens.


* Build the token-to-ID mappings directly from the raw tokenized text.


* **Causal Dataset Formatting:**
* Prepare input sequences using a sliding window approach of fixed length $N$ (e.g., length 16).


* Formulate standard next-token prediction targets by shifting sequences by one position:

$$\text{Input: } [t_0, t_1, \dots, t_{N-1}]$$


$$\text{Target: } [t_1, t_2, \dots, t_N]$$



---

### 3. Architecture Requirements (Decoder-Only Causal Transformer)

* **Remove Forced Head Masking & Roles:**
* Eliminate the artificial $W_{\text{window}}$ restrictions (`HEAD_ROLES` and `build_head_masks`).


* Allow all heads to learn representation subspaces freely without hardcoded local/long-range constraints.


* **Causal Self-Attention (Lower Triangular Mask):**
* Implement an additive causal mask (lower-triangular zero matrix with $- \infty$ in the upper triangle).


* Ensure query $Q_i$ at token position $i$ can only attend to key positions $K_j$ where $j \le i$.


* Apply scaled dot-product attention using this causal mask before computing softmax.


* **Embeddings & Positional Encodings:**
* Retain token embeddings and standard sinusoidal positional encodings.


* Ensure positional encodings scale dynamically with variable sequence lengths.


* **Output Projection:**
* Pass contextual attention outputs through a linear projection layer mapped to the vocabulary size to compute next-token logits.


---

### 4. Training & Optimization Requirements

* **Loss Function:**
* Use standard Cross-Entropy Loss over the unnormalized vocabulary logits across all sequence timesteps.


* Configure `ignore_index` to exclude padding tokens (`[PAD]`) from loss calculations.


* **Training Setup:**
* Train using Adam optimizer for 200 epochs.


* Log training loss periodically (e.g., every 40 epochs) to confirm convergence.


---

### 5. Evaluation, Inference & Diagnostic Requirements (Addressing Fixes)

* **Dynamic Generation Routine (Greedy/Top-k Sampling):**
* Include a auto-regressive generation function that accepts a prompt string and iteratively predicts tokens up to a max length.
* Ensure the model appends newly predicted tokens to the sequence and updates causal attention masks on each step.


* **Causal Witness Highlighter / Attention Visualization:**
* Update the attention inspector to display the lower-triangular causal attention row for target tokens.


* **Visualization Fix:** Dynamically scale ASCII visual weights based on the maximum attention score in the active row instead of fixed scaling, ensuring smaller attention values remain visible in console outputs.


* **Phrase Embedding & Anisotropy Fixes:**
* **Masked Mean Pooling:** Compute phrase embeddings by averaging contextual vectors over unpadded positions only, preventing padding vectors from skewing phrase averages.


* **Anisotropy Mean-Centering:** Compute the global mean vector across all token representations in the vocabulary and subtract it from phrase embeddings prior to calculating cosine similarity. This mitigates embedding space collapse and baseline similarity drift.


* **Heatmap / Positional Encoding Visualization:**
* Add a function or display routine showing sinusoidal positional encodings over both short ($N=16$) and long ($N=500$) sequence horizons to demonstrate low-frequency channel completion over longer contexts.
