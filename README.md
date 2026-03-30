# kv-collapse-profiler

**By [Christopher Lynch](https://github.com/clynch5) | [Intuitive Context LLC](https://intuitivecontext.com)** — AI consulting & applied ML research

Identify attention heads with geometrically collapsed key representations in any HuggingFace transformer model.

> **Full writeup:** [KV Collapse: Applied Research](https://intuitivecontext.com/blog/kv-collapse-applied-research)

## Quick Start

```bash
pip install transformers torch numpy
python kv_collapse_profiler.py --model Qwen/Qwen2.5-1.5B-Instruct
# JSON collapse map written to output/
```

## TLDR -- What This Does

One forward pass through a model. Measures the geometric spread of key vectors in every attention head. Heads where keys collapse to a narrow cone are candidates for aggressive quantization (1-2 bits) with zero quality loss. You get a per-head compression map in ~4 minutes on Apple Silicon, no GPU required.

## The Problem This Solves

KV cache compression today is uniform. [TurboQuant](https://arxiv.org/abs/2501.06225) (Google Research, 2025) applies the same bit-width across all heads using randomized Hadamard rotations. That works well as a baseline.

But not all heads are equal. Some heads have key vectors that cluster so tightly they carry almost no directional information. These heads can go to 1-2 bits without rotation, without calibration, without any quality loss at all. TurboQuant doesn't know which heads those are -- it treats every head the same.

This profiler tells you exactly which heads are collapsed and by how much. It's a layer on top of uniform compression: run TurboQuant everywhere, then drop the collapsed heads even further.

## What You Can Do With It TODAY

**Working now:**

- **Profile any HuggingFace model.** Point it at a model ID, get a JSON collapse map with per-head scores and classifications. Tested on Qwen2.5, SmolLM2, Qwen3, and others.
- **Use with [turboquant_plus](https://github.com/nickthecook/turboquant_plus)** -- a fork that runs TurboQuant on Apple Silicon. Profile first, then apply uniform compression to the rest.
- **Inform vLLM per-head FP8 calibration.** vLLM already supports per-head FP8 KV cache with calibration data. The collapse map tells you which heads to target.

**Coming soon:**

- **Per-head quantization in llama.cpp.** Today, llama.cpp only supports uniform `--cache-type-k` across all heads. Per-head allocation is expected Q2-Q3 2026. When it lands, your profile is ready -- the JSON output is designed for direct integration.

The honest state of the pipeline: the profiler works. The per-head allocation runtime doesn't exist yet in llama.cpp. This tool generates the map; the inference engines are catching up.

## Output

The profiler writes a JSON file with per-head collapse scores (0.0 = maximally spread, 1.0 = fully collapsed) and a classification for each head:

| Classification | Collapse (consistent_frac) | Meaning | Compression Guidance |
|---|---|---|---|
| `EXTREME` | >= 60% | Keys near-identical across inputs. | 1-2 bit. No rotation needed. |
| `HIGH` | 25-60% | Strong clustering. Limited directional entropy. | 2-4 bit. Rotation optional. |
| `MODERATE` | 10-25% | Mild signal. Some redundancy. | 4 bit. Standard quantization safe. |
| `NORMAL` | < 10% | Keys use the full representational space. | 8-16 bit. Full precision or standard quantization. |

Example entry from `collapsed_heads` array:

```json
{
  "layer": 15,
  "head": 1,
  "norm_depth": 0.536,
  "frac_pass1": 0.9804,
  "frac_pass2": 0.9784,
  "consistent_frac": 0.9784,
  "classification": "EXTREME",
  "recommended_bits": "1-2 bit",
  "val_frac_pass1": 0.0
}
```

## Results Across Three Architectures

| Model | Peak Collapse | Head | Architecture | Notes |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 98% | L15-H1 | Alibaba | Heavy collapse in mid-to-late layers |
| SmolLM2-1.7B-Instruct | 92% | L5-H21 | HuggingFace | Collapse concentrated in early layers |
| Qwen3-1.7B | 55% | L4-H0 | Alibaba (2025) | Newer architecture, less collapse overall |

Consistent finding across all models tested: **values never collapse. Only keys.** This is architecturally expected -- keys define the retrieval manifold and are more susceptible to geometric redundancy than values, which carry the content payload.

## How It Works

1. **Single forward pass** with a fixed prompt through the model, capturing all key states via hooks.
2. **L2-normalize** every key vector in every head across all layers.
3. **Compute pairwise minimum cosine distances** between key vectors within each head (causal mask applied -- each position only sees prior positions).
4. **Aggregate** the minimum distance statistics per head. Heads where the minimum distance is consistently near zero have collapsed keys -- all vectors point in roughly the same direction.
5. **Score and classify** each head on the 0-1 collapse scale.

No training data. No optimization steps. No GPU. Pure geometric measurement.

## Prior Art

This work is complementary to existing KV cache research:

- **[DuoAttention](https://arxiv.org/abs/2410.10819)** (MIT, ICLR 2025) -- Identifies streaming vs. retrieval heads. Requires ~2000 optimization steps on A100 GPUs with calibration data. Produces a binary streaming/retrieval classification.
- **[Loki](https://arxiv.org/abs/2406.02069)** (2024) -- Selective KV compression targeting less critical heads. Calibration-based.
- **[EigenAttention](https://arxiv.org/abs/2411.15245)** (2024) -- Spectral analysis of attention patterns for compression decisions. Training-dependent.
- **[Inheritune](https://arxiv.org/abs/2404.01413)** (2024) -- Layer-dropping approaches to efficient fine-tuning.

**What's different here:** No calibration. No GPU. No training data. A geometric probe that any practitioner can run on consumer hardware in minutes. The contribution isn't a new compression algorithm -- it's the applied gap. Nobody has shipped a tool that gives you actionable per-head compression maps this cheaply.

## Applying the Results

### vLLM (works today)

vLLM supports per-head FP8 KV cache quantization. Use the collapse map to identify which heads to target for aggressive compression during calibration. Collapsed heads can skip calibration entirely.

### llama.cpp (uniform only, per-head coming)

Current state: `--cache-type-k` applies one quantization type to all heads. Per-head support is tracked upstream and expected Q2-Q3 2026. The JSON output from this profiler is structured for direct integration when that lands.

### TurboQuant integration

Use with [turboquant_plus](https://github.com/nickthecook/turboquant_plus) for a two-stage strategy:

1. **Profile** -- Run this tool to identify collapsed heads.
2. **Compress uniformly** -- Apply TurboQuant (Hadamard rotation + uniform quantization) to all heads.
3. **Drop collapsed heads further** -- Heads scoring >= 0.90 can go to 1-2 bits without the rotation step, saving both memory and compute.

## License

MIT
