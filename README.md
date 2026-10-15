# Decoder-Only Transformer Language Model from Scratch (TensorFlow / Keras 3)

A GPT-style causal language model built from first principles in TensorFlow/Keras 3 and trained on WikiText-103 with the GPT-2 tokenizer. Every component (token embedding, positional encoding, multi-head self-attention, feed-forward block, decoder layer and the causal mask) is written by hand rather than imported from a library. The aim is to understand how a modern LLM works internally, not to compete with pretrained models.

---

## 1. The idea

A language model learns the probability of the next token given all the tokens before it:

```
P(x_1, ..., x_T) = Π_t P(x_t | x_1, ..., x_{t-1})
```

A **decoder-only transformer** models this directly:

1. Text is split into subword tokens (GPT-2 byte-pair encoding, vocabulary 50,257).
2. Each token becomes a vector (**embedding**), and its position is added (**positional encoding**), because attention on its own has no sense of order.
3. A stack of **decoder layers** lets each position gather information from earlier positions through **self-attention**, then transforms it with a **feed-forward network**.
4. A **causal (look-ahead) mask** blocks position *t* from seeing positions after *t*, so the model cannot cheat by reading the answer.
5. A final linear layer maps each position's vector to a score for every vocabulary token. Training minimises cross-entropy between these scores and the actual next token.

Training on input `x[0:T]` with target `x[1:T+1]` gives *T* next-token predictions per sequence in a single forward pass. This parallelism is the main reason transformers replaced RNNs.

### Scaled dot-product attention

```
Attention(Q, K, V) = softmax( (Q Kᵀ) / √d_k + M ) V
```

`M` is 0 where attention is allowed and −10⁹ above the diagonal, so those weights become ~0 after the softmax. **Multi-head** attention runs this in parallel in `h` lower-dimensional subspaces (`d_k = d_model / h`) and concatenates the results, so different heads can track different relationships.

---

## 2. Architecture

```
token ids (B, T)
   │
TokenEmbedding  ── Embedding(vocab, d_model) × √d_model
   │
PositionalEncoding ── + fixed sinusoidal table (max_len 5000)
   │
Dropout
   │
┌──────────────── DecoderLayer × N ────────────────┐
│  x ─► MultiHeadAttention(x, x, x, causal mask)   │
│        └► Dropout ─► Add(x) ─► LayerNorm  = out1 │
│  out1 ─► FeedForward (Dense dff, ReLU, Dense)    │
│        └► Dropout ─► Add(out1) ─► LayerNorm      │
└──────────────────────────────────────────────────┘
   │
Dense(vocab)  ── logits (B, T, vocab)
```

This is the **post-norm** layout of the original *Attention Is All You Need* paper (LayerNorm after the residual add).

### Default configuration

| Hyperparameter | Value |
|---|---|
| Layers (`NUM_LAYERS`) | 6 |
| Model width (`D_MODEL`) | 512 |
| Attention heads (`NUM_HEADS`) | 8 (head dim 64) |
| FFN inner width (`DFF`) | 2048 |
| Dropout | 0.1 |
| Context length (`SEQ_LENGTH`) | 512 tokens |
| Batch size | 4 sequences (2,048 tokens / step) |
| Optimizer | Adam, constant LR 1e-4 |
| Vocabulary | 50,257 (GPT-2 BPE) |

**Parameter count: ~70.4M.**

| Block | Parameters |
|---|---|
| Token embedding (50,257 × 512) | 25.7M |
| 6 decoder layers (~3.15M each) | 18.9M |
| Output projection (512 × 50,257 + bias) | 25.8M |

About 73% of the parameters are in the input and output vocabulary matrices. Tying them (see §7) is the cheapest size reduction available.

---

## 3. Project structure

```
transformer_lm.ipynb
├── 1. Setup           – installs, imports, GPU check
├── 2. Data pipeline   – WikiText-103 → GPT-2 tokens → (input, target) windows → tf.data
├── 3. Model           – TokenEmbedding, PositionalEncoding, MultiHeadAttention,
│                        FeedForward, DecoderLayer, Transformer
├── 4. Build & compile – instantiate, compile, model.summary()
├── 5. Train           – model.fit with a fixed step budget, loss plot
└── 6. Sample          – top-k / temperature text generation
```

---

## 4. How to implement it (step by step)

### 4.1 Environment

```bash
pip install tensorflow datasets transformers numpy matplotlib
```

Tested with TensorFlow 2.18–2.21 and Keras 3. A GPU is strongly recommended; the Colab T4 free tier works at the default batch size.

### 4.2 Data pipeline

1. Load `Salesforce/wikitext`, config `wikitext-103-raw-v1` (~1.8M lines in train).
2. Stream each non-empty line through `GPT2Tokenizer.encode` into a running buffer.
3. Whenever the buffer holds ≥ `SEQ_LENGTH + 1` tokens, emit:
   - `input  = buffer[0 : SEQ_LENGTH]`
   - `target = buffer[1 : SEQ_LENGTH + 1]`

   then advance the buffer by `SEQ_LENGTH`.
4. Wrap the generator in `tf.data.Dataset.from_generator`, `.repeat()` the training stream, shuffle with a 1,000-sequence buffer, then `.batch()` and `.prefetch()`. Validation is neither shuffled nor repeated.

Lines are concatenated across article boundaries with no separator token. That is standard for this kind of LM training, but it means some windows straddle two unrelated articles.

### 4.3 Model

Implement each block as a `keras.layers.Layer` subclass. Follow these Keras 3 rules, which caused the crash in the earliest version of this code:

- **Pass only tensors positionally.** Calling `layer(x, training, mask)` raises `ValueError: Only input tensors may be passed as positional arguments`. Use `layer(x, training=training, look_ahead_mask=mask)`.
- **Don't call the causal-mask argument `mask`.** Keras treats `mask` as its own padding-mask mechanism and warns that the layer "does not support masking". The mask still reaches `call()`, but naming it `look_ahead_mask` keeps it clearly separate from Keras's masking.
- Keep a `training` argument on every layer that contains `Dropout`. Keras 3 passes the outer `training` value down automatically, but writing it explicitly makes the intent clear.

Causal mask:

```python
mask = 1 - tf.linalg.band_part(tf.ones((T, T)), -1, 0)   # 1 above diagonal
logits += mask[None, None] * -1e9                        # broadcast over batch & heads
```

### 4.4 Training

```python
model.compile(optimizer=Adam(1e-4),
              loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(train_dataset, epochs=3, steps_per_epoch=500,
          validation_data=val_dataset, validation_steps=50)
```

**Know the budget.** WikiText-103 is roughly 120M GPT-2 tokens (an estimate; the exact count depends on tokenisation). At 2,048 tokens per step, one true epoch is about 58k steps. The default of 500 × 3 = 1,500 steps covers about **2.5% of a single epoch**. It shows that training works and the loss falls; it does not produce a usable model. Expect samples to stay incoherent until tens of thousands of steps.

Track **perplexity = exp(validation loss)**. A uniform guess over the vocabulary gives a loss of ln(50,257) ≈ 10.8, so the first epoch should start well below that and keep dropping.

### 4.5 Sampling

Autoregressive generation with temperature and top-k:

1. Encode the prompt.
2. Feed the last ≤ 512 tokens and take the logits at the final position.
3. Divide by `temperature`, keep the top-k, softmax, and sample one token.
4. Append it and repeat.

Each step re-runs the full context because there is no KV cache, so generation is O(T²) per token. That's fine for demos and slow for long outputs.

### 4.6 Verifying correctness

Cheap tests worth keeping:

- **Causality:** change the token at position *p*. Logits at positions < *p* must not change.
- **Determinism:** two forward passes with `training=False` must give identical outputs (dropout off).
- **Overfit one batch:** loss on a single repeated batch should approach ~0. If it doesn't, something is wired wrong.
- **Initial loss:** should be close to ln(vocab) ≈ 10.8 before training.

---

## 5. Applications

What a model like this is useful for **once it is trained adequately**, roughly from most to least realistic at this size:

- **Learning and teaching.** The main value of this project: seeing exactly how attention, masking, residuals and normalisation fit together, and being able to change any of them.
- **Architecture experiments.** A controlled testbed to compare pre-norm vs post-norm, learned vs sinusoidal vs rotary positions, weight tying, LR schedules and so on, at a scale that fits on one GPU.
- **Domain-specific language models.** Train the same code on a narrow corpus (technical standards, legal text, logs, code) for autocomplete, or as a starting point for domain fine-tuning.
- **Text scoring.** Perplexity under the model can rank or filter text, flag out-of-distribution inputs, or score candidate outputs from another system.
- **Feature extraction.** Hidden states from the final decoder layer can serve as contextual embeddings for downstream classifiers. Encoder-style models are usually better at this.
- **Text generation and completion.** Possible, but at 70M parameters trained on WikiText alone, expect locally fluent and factually unreliable text.

What it is **not** suited for: instruction following, chat, factual question answering or anything production-facing. Those require far more data and compute, plus instruction tuning and alignment stages this project does not include.

---

## 6. Known limitations and pitfalls

- **Mixed precision does not work as written.** Enabling `mixed_float16` makes the first forward pass fail with `InvalidArgumentError`. `TokenEmbedding` (the √d scale), `PositionalEncoding` (the float32 table) and the attention scaling and mask all combine float32 constants with float16 tensors. Cast each with `tf.cast(..., x.dtype)` before enabling it, and use a smaller mask constant such as `-1e4`, since `-1e9` overflows float16.
- **Post-norm** is less stable than pre-norm as depth increases. It is fine at 6 layers.
- **Constant learning rate** with no warmup. It works, but converges more slowly and less reliably than warmup plus decay.
- **No checkpointing.** A Colab disconnect loses training progress. Add `keras.callbacks.ModelCheckpoint`.
- **Weak shuffling.** A 1,000-sequence buffer over a stream that follows document order means neighbouring batches come from similar articles.
- **No KV cache**, so sampling is slow for long generations.

---

## 7. Roadmap

Roughly ordered by payoff per unit of effort:

1. **Warmup + cosine decay** learning-rate schedule.
2. **Pre-norm** blocks: `x + Attn(LN(x))`, then a final LayerNorm before the output layer.
3. **Weight tying** between the token embedding and the output projection (−25.7M parameters, ~37%).
4. **Checkpointing and perplexity logging** per epoch.
5. **Mixed precision** (after the dtype fixes in §6) to roughly halve memory use and speed up training on tensor-core GPUs.
6. **Learned or rotary positional embeddings** in place of the fixed sinusoidal table.
7. **GELU** activation and a **KV cache** for generation.
8. Larger step budget, then evaluation on the WikiText-103 test split.

---

## 8. References

- Vaswani et al., *Attention Is All You Need*, 2017. The original transformer.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019. The decoder-only LM setup and the BPE tokenizer.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020. Pre-norm vs post-norm.
- Press & Wolf, *Using the Output Embedding to Improve Language Models*, 2017. Weight tying.
- Merity et al., *Pointer Sentinel Mixture Models*, 2016. Introduces WikiText-103.
