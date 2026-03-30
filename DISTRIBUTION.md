# KV Collapse Profiler — Launch Distribution Content

Ready-to-publish content for each channel. Copy, paste, post.

---

## 1. LinkedIn Post

Most attention heads in small language models are doing almost nothing with their keys.

I built a profiler that measures geometric collapse in KV cache key representations across every attention head in a HuggingFace model. Single forward pass. No GPU. No training data. About 4 minutes on a MacBook.

The results are striking. Qwen2.5-1.5B has a head (L15-H1) at 98% key collapse — nearly every key vector is a near-duplicate of a previous one, regardless of input. SmolLM2-1.7B hits 92%. Qwen3-1.7B is 55%. Values never collapse. Only keys. This pattern holds across every architecture tested.

Why it matters: tools like TurboQuant (Google) compress uniformly at 3-4 bits per head. This profiler identifies heads where 1-2 bits is sufficient. DuoAttention (MIT) does similar identification but requires 2000 optimization steps on A100 GPUs. This runs on consumer hardware.

The per-head compression pipeline in llama.cpp isn't there yet — that's coming Q2-Q3 2026. The profiler works today as a structural diagnostic.

Open source. Try it on your own models.

https://github.com/clynch5/kv-collapse-profiler
https://intuitivecontext.com/blog/kv-collapse-applied-research

---

## 2. Hacker News Submission

**Title:** Show HN: KV Collapse Profiler — find attention heads that waste bits in any LLM

**Author Comment:**

I built a tool that profiles geometric collapse in KV cache key representations across every attention head in a HuggingFace transformer model. It runs a single forward pass over ~900 tokens, computes cosine similarity between consecutive key vectors, and flags heads where most keys are near-duplicates of previously seen vectors regardless of input content.

Results on three architectures: Qwen2.5-1.5B has a head at 98% key collapse (L15-H1), SmolLM2-1.7B hits 92%, Qwen3-1.7B at 55%. The consistent finding across all models tested: values never collapse, only keys. This asymmetry is well-documented in the literature (Loki 2024, EigenAttention 2024, lazy layers/Inheritune 2024) but there wasn't a practical per-head diagnostic tool that a practitioner could point at an arbitrary model.

Prior art I want to be explicit about: DuoAttention (MIT, ICLR 2025) identifies head specialization patterns for KV cache optimization. Their approach requires 2000 optimization steps on A100 GPUs with synthetic training data. This profiler does something narrower — it only detects geometric collapse, not the full specialization taxonomy — but it does it in 4 minutes on a MacBook with no training data and no calibration. Complementary, not competitive.

The practical application is informing mixed-precision KV cache quantization. TurboQuant (Google) compresses KV caches but applies uniform bit allocation across heads. This profiler produces the per-head collapse map that tells you which heads can safely go to 1-2 bits while TurboQuant spends 3-4 bits everywhere.

Honest status: the profiler works today and produces actionable JSON profiles. The actual per-head mixed-precision pipeline in llama.cpp/vLLM/etc. doesn't exist yet. That integration is the hard part, and I expect it to land Q2-Q3 2026. Right now this is a diagnostic and research tool. The gap between "we know which heads are collapsed" and "the inference engine acts on that knowledge" is real.

Python, runs on CPU, ~4 minutes for a 1.5B model on M-series MacBook.

https://github.com/clynch5/kv-collapse-profiler

---

## 3. Reddit r/LocalLLaMA Post

**Title:** I built a tool that finds which attention heads in your model are wasting memory — runs in 4 min on a MacBook, no GPU

**Body:**

I kept reading papers about KV cache compression and noticed a gap: plenty of research showing that key representations in certain attention heads collapse to near-duplicates, but no simple tool that lets you profile this on your own models without a GPU cluster.

So I built one. Point it at any HuggingFace model, it runs a single forward pass on CPU, and produces a per-head collapse map telling you exactly which heads have redundant key vectors.

**What I found testing 1-2B models:**

| Model | Worst Head | Collapse % |
|-------|-----------|------------|
| Qwen2.5-1.5B | L15-H1 | 98% |
| SmolLM2-1.7B | — | 92% |
| Qwen3-1.7B | — | 55% |

The pattern across every architecture: values never collapse, only keys. This is consistent with published research (Loki, EigenAttention, lazy layers) but nobody had packaged it as a practitioner tool.

**Why this matters for the local LLM crowd:**

If you're running models through Ollama, llama.cpp, or LM Studio, the KV cache is eating your RAM. Current quantization (GGUF Q4, Q5, etc.) compresses weights but the KV cache stays in full precision or gets uniform quantization. This profiler identifies heads where the keys could safely compress to 1-2 bits instead of 3-4 bits — potentially significant memory savings during inference.

**Hardware requirements:** Tested on an M3 MacBook Pro. About 4 minutes for a 1.5B model. No GPU needed. No training data. No calibration step. Just `pip install torch transformers numpy` and run.

**What this is NOT (yet):** The profiler produces the per-head collapse map. The actual per-head mixed-precision pipeline in llama.cpp doesn't exist yet. That integration work is coming (targeting Q2-Q3 2026). Right now, the profiler is a diagnostic tool — it tells you WHERE to compress, but the serving engines don't yet support compressing each head independently.

For context: DuoAttention (MIT) does similar head identification but requires 2000 optimization steps on A100 GPUs. Google's TurboQuant compresses KV caches but doesn't differentiate between heads. This sits in the gap between those two.

Open source, MIT license. Try it on whatever model you're running locally and share your results — I'm genuinely curious how the collapse patterns vary across model families.

https://github.com/clynch5/kv-collapse-profiler

Full writeup: https://intuitivecontext.com/blog/kv-collapse-applied-research

---

## 4. Reddit r/MachineLearning Post

**Title:** [P] KV Collapse Profiler: calibration-free per-head geometric collapse detection for KV cache compression

**Body:**

Sharing a tool that operationalizes known findings about key vector low-rank structure into a practical per-head diagnostic.

**Background:** Multiple recent works document that key representations in certain attention heads exhibit geometric collapse — near-duplicate vectors regardless of input diversity. Loki (2024) and EigenAttention (2024) characterize low-rank key structure. Lazy layers / Inheritune (2024) identify redundant computation in specific layers. DuoAttention (Xiao et al., ICLR 2025) classifies heads by specialization pattern using 2000 optimization steps on A100 GPUs with synthetic calibration data.

**The gap:** No lightweight, calibration-free tool existed to produce a per-head collapse map on commodity hardware. Practitioners experimenting with mixed-precision KV cache quantization (motivated by TurboQuant and similar work) lack the structural diagnostic to identify WHERE non-uniform bit allocation is safe.

**This tool:** Single forward pass over ~900 diverse tokens. Measures cosine similarity between consecutive key and value vectors per head. Flags heads exceeding a configurable collapse threshold. Produces JSON output with per-head scores and compression recommendations. Runs in ~4 minutes on an M-series MacBook for 1.5B parameter models. No GPU, no training data, no calibration.

**Findings across architectures:** Key collapse rates of 98% (Qwen2.5-1.5B L15-H1), 92% (SmolLM2-1.7B), and 55% (Qwen3-1.7B). Value collapse is consistently near zero across all models tested, supporting the key-specific nature of the phenomenon.

**Limitations:** This detects geometric collapse only, not the broader head specialization taxonomy that DuoAttention captures. The downstream per-head mixed-precision pipeline in production inference engines (llama.cpp, vLLM) is not yet available — this produces the map, not the compression. Engine integration is expected Q2-Q3 2026.

Code: https://github.com/clynch5/kv-collapse-profiler

Writeup: https://intuitivecontext.com/blog/kv-collapse-applied-research

---

## 5. Twitter/X Thread

**1/5**
Most attention heads barely use their keys.

I profiled KV cache collapse across every head in Qwen2.5-1.5B, SmolLM2-1.7B, and Qwen3-1.7B. One head hit 98% key collapse — nearly every key vector is a near-duplicate regardless of input. Values never collapse. Only keys. Every architecture.

**2/5**
Why it matters: TurboQuant (Google) compresses KV caches but treats every head the same at 3-4 bits. Some heads only need 1-2 bits. The profiler identifies which ones. DuoAttention (MIT) does similar identification but needs 2000 optimization steps on A100 GPUs. This takes 4 minutes on a MacBook.

**3/5**
No GPU. No training data. No calibration. Single forward pass. Point it at any HuggingFace model, get a per-head collapse map with compression recommendations. Tested on Apple Silicon — about 4 minutes for a 1.5B model.

**4/5**
Being honest about status: the profiler works today and produces actionable profiles. The per-head mixed-precision pipeline in llama.cpp/vLLM doesn't exist yet — that's the hard part, targeting Q2-Q3 2026. Right now this is the diagnostic that those engines will eventually need.

**5/5**
Open source. MIT license. Try it on your own models.

Code: github.com/clynch5/kv-collapse-profiler
Writeup: intuitivecontext.com/blog/kv-collapse-applied-research
