# Directed Reading: GPU Programming for Machine Learning Acceleration

**Duration:** 12 weeks · ~10–12 hrs/week (readings + mini-project)  
**Prerequisites:** C/C++ proficiency, basic computer architecture (caches, pipelines), linear algebra, PyTorch fluency, understanding of backpropagation and standard neural network layers  
**Hardware:** Access to an NVIDIA GPU (Ampere or newer recommended; Hopper ideal)

---

## Course Philosophy

Machine learning workloads are dominated by a small number of computational patterns — dense matrix multiplication, elementwise fusions, reductions, softmax, and attention — but extracting peak performance from each requires deep understanding of GPU hardware. This course builds GPU programming fluency from architecture fundamentals through production-grade kernel engineering, following the path a researcher or ML systems engineer actually walks: from writing a first kernel, to understanding why GEMM is the way it is, to fusing custom ops, to writing attention kernels and understanding the full training stack. Each week pairs foundational GPU reading with a hands-on mini-project that targets a real ML primitive.

---

## Phase 1: GPU Architecture & CUDA Fundamentals (Weeks 1–4)

### Week 1 — The GPU Execution Model

**Goal:** Understand the GPU as a throughput-oriented machine and write your first kernels.

**Readings**

1. [A Brief History of GPUs](https://fabiensanglard.net/cuda/) — Evolution from graphics pipelines to general-purpose compute; why the hardware is shaped the way it is.
2. [GPU Glossary — Device Hardware](https://modal.com/gpu-glossary/device-hardware/tensor-core) — Reference for SM, warp, register file, and tensor core terminology. Skim the full glossary; bookmark for later.
3. CUDA C Programming Guide, Chapters 1–3 (NVIDIA docs) — Programming model (grids, blocks, threads), memory spaces, kernel launch syntax.

**Supplementary**

- NVIDIA GTC talk: [Maximize Bandwidth and Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/) — First 20 minutes cover the execution model at a high level.

**Mini-Project 1 — Vector Addition → Fused Elementwise Kernel**

Start with the canonical vector-add kernel. Then extend it to a fused elementwise kernel that computes `y = GELU(x * W_scale + bias)` for a 1D tensor — the post-linear activation pattern that appears everywhere in transformers. Profile with `nsys`, measure achieved bandwidth vs. peak, and vary tensor length from 1K to 16M elements.

*Why this matters for ML:* Elementwise fusions are the first thing you write custom kernels for in practice. Understanding memory-bound vs. compute-bound starts here.

---

### Week 2 — Memory Hierarchy: Global, Shared, Registers

**Goal:** Internalize the GPU memory hierarchy and learn to exploit shared memory for data reuse.

**Readings**

1. CUDA C Programming Guide, Chapter 5 (Performance Guidelines) — Coalescing, occupancy, shared memory usage patterns.
2. [Shared Memory Microbenchmarks](https://feldmann.nyc/blog/smem-microbenchmarks) — Empirical measurements of bank conflicts, padding strategies, and throughput.
3. [Making Matrix Transpose Really Fast on Hopper](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) — Banking, layout, and conflict-free access patterns through the lens of a deceptively simple kernel.

**Supplementary**

- CUDA C Programming Guide, Appendix C (warp-level intrinsics) — Reference for `__shfl_*` and warp-level reductions.

**Mini-Project 2 — Tiled Matrix Transpose + Softmax**

Implement (a) a shared-memory tiled matrix transpose that avoids bank conflicts, then (b) a numerically stable softmax kernel (subtract max, exponentiate, divide by sum) that uses shared memory for the reduction steps. Softmax operates row-wise on a matrix of shape (B, V) where V = vocabulary size (32K–128K). Benchmark against PyTorch's `F.softmax`.

*Why this matters:* Softmax is in every attention layer and at every output head. It's memory-bound, reduction-heavy, and a perfect vehicle for learning shared memory patterns.

---

### Week 3 — Warp-Level Programming & Reductions

**Goal:** Move from block-level to warp-level thinking; master reductions, scans, and online algorithms.

**Readings**

1. [CUDA GEMM — Step by Step](https://siboehm.com/articles/22/CUDA-MMM) (Sections 1–6 only) — Not for GEMM itself yet, but for its clear exposition of tiling, register blocking, and occupancy tuning.
2. [Register-Level Tiling](https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html) — Algorithmic basics of how data moves from global → shared → registers.
3. Mark Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA whitepaper, classic) — The canonical walk through 7 reduction kernels from naïve to warp-shuffle.

**Supplementary**

- CUB library documentation (`BlockReduce`, `BlockScan`, `WarpReduce`).
- Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018) — The online softmax trick used in FlashAttention.

**Mini-Project 3 — Online Softmax + LayerNorm**

Implement (a) an online softmax kernel using the Milakov–Gimelshein algorithm (single-pass: track running max and running sum simultaneously), and (b) a fused LayerNorm kernel (mean, variance, normalize, scale, shift in one pass). Compare your online softmax against Week 2's two-pass version. Benchmark against PyTorch and `apex.normalization.FusedLayerNorm`.

*Why this matters:* Online algorithms are the foundation of FlashAttention. LayerNorm fusion is one of the highest-impact custom kernels in transformer training.

---

### Week 4 — Streams, Concurrency & Profiling

**Goal:** Learn to overlap compute, memory transfers, and kernel launches; become proficient with NVIDIA profiling tools.

**Readings**

1. CUDA C Programming Guide, Chapter 3.2.8 (Asynchronous Concurrent Execution) — Streams, events, default stream behavior.
2. [CUDA Graphs — Accelerating Generative AI](https://pytorch.org/blog/accelerating-generative-ai-2/) — How CUDA graphs eliminate launch overhead for repeated inference workloads.
3. Nsight Systems and Nsight Compute documentation (quick-start guides) — Learn to read a timeline, identify stalls, and interpret occupancy/warp schedulers.

**Supplementary**

- PyTorch internals: `torch.cuda.Stream`, `torch.cuda.graph` API documentation.

**Mini-Project 4 — Profiling a Transformer Block End-to-End**

Take a standard PyTorch transformer encoder block (self-attention + FFN + LayerNorm + residuals) and profile it end-to-end with Nsight Systems. Identify: which kernels dominate wall-clock time, where kernel launch overhead is significant, where memcpy stalls appear. Then wrap the block in a CUDA graph and measure latency reduction. Produce an annotated Nsight timeline as your deliverable.

*Why this matters:* Before you optimize anything, you need to know where the time goes. This project builds the diagnostic skills that inform every decision in Weeks 5–12.

---

## Phase 2: The GEMM–Attention Axis (Weeks 5–8)

### Week 5 — Dense GEMM: The Kernel That Runs ML

**Goal:** Understand GEMM as the canonical GPU kernel and the computational backbone of every linear layer, attention projection, and MLP.

**Readings**

1. [CUDA GEMM — Step by Step](https://siboehm.com/articles/22/CUDA-MMM) (complete) — Full walk-through from naïve to near-cuBLAS performance.
2. [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — Real engineering worklog pushing GEMM on Hopper; exposes the gap between textbook and production.
3. [CuTE Layouts](https://yang-yifan.github.io/blogs/cute_layout/cute_layout.html) — CUTLASS's layout abstraction; how logical tile coordinates map to physical memory.

**Supplementary**

- CUTLASS documentation: GemmUniversal interface and tiled-GEMM concepts.
- Roofline model primer (if unfamiliar): Williams, Waterman & Patterson, "Roofline" (2009).

**Mini-Project 5 — GEMM From Scratch, 6 Versions**

Following Siboehm's progression, implement 6 versions of SGEMM: (1) naïve, (2) global memory coalescing, (3) shared memory tiling, (4) 1D register blocktiling, (5) 2D register blocktiling, (6) vectorized loads (`float4`). Benchmark each against cuBLAS for M=N=K=4096. Plot TFLOPS and produce a roofline diagram positioning each version.

*Why this matters:* GEMM constitutes 60–80% of training FLOPS in a transformer. Understanding its optimization hierarchy is non-negotiable for anyone doing ML systems work.

---

### Week 6 — Tensor Cores & MMA Programming

**Goal:** Understand hardware matrix-multiply units and how to program them at the PTX and WMMA levels.

**Readings**

1. [Tensor Core Evolution: Volta to Blackwell](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/) — Architecture evolution, data formats (FP16, TF32, FP8, INT8), and throughput scaling across generations.
2. [WMMA and MMA Programming](https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master/src/wmma) + [MMA PTX Notes](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d) — Hands-on code for warp-level matrix operations.
3. [Fast GEMM with Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) — From-scratch HGEMM using tensor cores.

**Supplementary**

- NVIDIA Mixed Precision Training whitepaper (Micikevicius et al., 2018) — Loss scaling, FP16 master weights, where reduced precision is safe.

**Mini-Project 6 — Tensor-Core HGEMM + Precision Analysis**

Implement an FP16 GEMM using WMMA intrinsics (16×16×16 tiles). Then extend to TF32 (same kernel structure, different MMA instruction). For a realistic transformer linear layer shape (e.g., 4096 × 4096 × 11008 for a LLaMA FFN), measure: (a) TFLOPS achieved vs. peak TC throughput, (b) numerical error relative to FP32 cuBLAS across 1000 random matrices, (c) the effect of loss scaling on gradient accuracy in a toy training loop.

*Why this matters:* Mixed precision training is standard practice. Understanding the hardware data path and where precision loss occurs — at the accumulator, at the storage, at the reduction — is critical for debugging training instabilities.

---

### Week 7 — Advanced Memory: TMA & Async Copy (Hopper+)

**Goal:** Learn Hopper/Blackwell's Tensor Memory Accelerator and asynchronous bulk copy for decoupling compute from data movement.

**Readings**

1. [TMA Basics with CuTE](https://yang-yifan.github.io/blogs/cute_tma/cute_tma.html) — How TMA descriptors work, address generation, and multicast.
2. [Blackwell GEMM with CUTLASS](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) — Tensor Memory (TMEM) and 5th-generation tensor cores.
3. [Persistent Dynamic Launch (PDL)](https://yang-yifan.github.io/blogs/pdl/pdl.html) — Dynamic kernel scheduling for persistent kernels.

**Supplementary**

- Code repository: [Yang-YiFan blog code](https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs)
- CUTLASS 3.x source: `cute/arch/copy_sm90_tma.hpp` — Read the TMA copy atoms.

**Mini-Project 7 — Software-Pipelined GEMM Tile Loader**

Implement (or extend your Week 6 kernel with) a double-buffered software pipeline: while one shared-memory buffer is consumed by MMA instructions, the other is being filled via `cp.async` (Ampere) or TMA (Hopper). Measure the compute-to-memory overlap ratio in Nsight Compute. If Hopper hardware is unavailable, implement the `cp.async` version and write a 1-page design document describing how TMA would change the implementation.

*Why this matters:* Software pipelining is how FlashAttention, CUTLASS, and every production GEMM kernel hide memory latency. This is the technique, not a technique.

---

### Week 8 — FlashAttention: Tiling Meets Attention

**Goal:** Understand FlashAttention as the synthesis of everything learned so far — tiling, online algorithms, register-level data reuse — applied to the most important ML kernel.

**Readings**

1. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022) — The foundational paper. Focus on the tiled online softmax algorithm and the IO-complexity analysis.
2. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024) — Improved parallelism (sequence-parallel across warps), reduced non-matmul FLOPS.
3. Dao et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024) — Hopper-specific: warp specialization, FP8 attention, TMA integration.

**Supplementary**

- FlashAttention source code: `flash_attn/flash_attn_triton_amd.py` (Triton version is more readable) and `csrc/flash_attn/` (CUDA version).
- Blog: "From Online Softmax to FlashAttention" (various authors, multiple good explainers exist).

**Mini-Project 8 — Simplified FlashAttention (Forward Pass)**

Implement the forward pass of FlashAttention-1 for a single head: given Q, K, V of shape (N, d) with d = 64, compute the exact attention output using tiled online softmax (no materialization of the N×N attention matrix). Use shared memory for K/V tiles, registers for the running softmax state. Benchmark against (a) naïve PyTorch `torch.matmul(Q, K.T) → softmax → matmul(_, V)` and (b) `torch.nn.functional.scaled_dot_product_attention` with FlashAttention backend. Report memory savings and wall-clock speedup for N = 1K, 4K, 16K.

*Why this matters:* FlashAttention is arguably the most impactful ML systems paper of the 2020s. Implementing it yourself is the single best exercise for understanding GPU memory-hierarchy optimization in a real-world context.

---

## Phase 3: ML Systems & Production Kernels (Weeks 9–12)

### Week 9 — Triton: Rapid Kernel Development

**Goal:** Explore Triton as a high-productivity alternative to CUDA for custom ML kernels; understand its compilation model and when it matches or falls short of hand-tuned CUDA.

**Readings**

1. [Triton Language Documentation](https://triton-lang.org/) — Tutorial, language reference, and compilation model (tile-based programming, auto-tuning).
2. Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (MAPL 2019) — The academic paper behind Triton.
3. [Triton vs. Modular](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls) — Tradeoffs between embedded DSLs and standalone compilers.

**Supplementary**

- [Modular Platform](https://www.modular.com/) — Mojo language, MAX engine overview.
- Triton source: `python/tutorials/` — Walk through the matmul and fused-softmax tutorials.

**Mini-Project 9 — Triton Fused Attention + Autotuning**

Re-implement your Week 8 FlashAttention forward pass in Triton. Use `@triton.autotune` to search over block sizes for Q, K, V tiles and `num_warps`. Compare performance against your CUDA version and PyTorch's SDPA backend. Then implement a fused attention + RoPE kernel (apply rotary position embeddings inside the attention loop, before the QK dot product). Write a 1-page analysis: what Triton makes easier, what it makes harder, where you hit expressiveness limits.

*Why this matters:* In practice, most custom ML kernels at startups and research labs are written in Triton, not CUDA. Knowing both and understanding the tradeoff is how you make engineering decisions about when to drop to CUDA.

---

### Week 10 — Custom Ops for Training: Backward Passes & Autograd Integration

**Goal:** Write CUDA/Triton kernels that integrate into PyTorch's autograd for training, not just inference.

**Readings**

1. PyTorch documentation: "Extending PyTorch — Custom C++/CUDA Extensions" and `torch.autograd.Function`.
2. PyTorch documentation: "Custom Operators" (the `torch.library` API for `torch.compile`-compatible custom ops).
3. Dao et al., FlashAttention paper — Re-read Section 3.2 on the backward pass: recomputation of attention matrix from stored O, l, m instead of saving the N×N matrix.

**Supplementary**

- Triton documentation on writing backward-pass kernels.
- Liger-Kernel or Unsloth source code — Examples of production fused training kernels.

**Mini-Project 10 — Fused Cross-Entropy with Custom Backward**

Implement a fused cross-entropy loss kernel (logits → log-softmax → NLL loss → scalar, without materializing the full softmax output) in Triton or CUDA, with a custom backward pass that computes gradients w.r.t. logits directly. Register it as a `torch.autograd.Function`. Train a small GPT-2 model (124M) for 1000 steps with your custom loss vs. `F.cross_entropy` and verify: (a) gradient correctness (compare against PyTorch within tolerance), (b) memory savings (the softmax output for vocab_size=50K × batch is never materialized), (c) wall-clock speedup.

*Why this matters:* The fused cross-entropy pattern is used in every efficient training framework (Liger, Unsloth, Megatron). Writing the backward pass yourself demystifies autograd integration and teaches you what "recomputation vs. materialization" means concretely.

---

### Week 11 — Inference Optimization: Quantization, KV-Cache & Speculative Decoding

**Goal:** Understand the GPU kernel-level techniques that make LLM inference fast and memory-efficient.

**Readings**

1. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (NeurIPS 2022) — Mixed-precision decomposition, outlier features, INT8 matmul.
2. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (ICLR 2023) — Layer-wise quantization, Hessian-based rounding.
3. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023) — Draft-then-verify parallel decoding.

**Supplementary**

- vLLM source code: PagedAttention kernel and KV-cache manager.
- NVIDIA TensorRT-LLM documentation: INT8/FP8 kernels, inflight batching.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023).

**Mini-Project 11 — W4A16 Dequantize-Fused GEMM Kernel**

Implement a Triton or CUDA kernel that performs weight-only INT4 quantization with FP16 activations (W4A16): dequantize INT4 weights on-the-fly during the GEMM, using group-wise scale factors (group size = 128). Benchmark against (a) FP16 cuBLAS GEMM and (b) a naïve dequantize-then-GEMM approach, for a LLaMA-7B-shaped linear layer (4096 × 11008). Report: throughput (tokens/sec), memory savings, and output quality (compare perplexity on a small eval set if time permits).

*Why this matters:* W4A16 GEMM is the kernel that makes 7B models run on consumer GPUs. The dequantize-fused pattern (never materializing FP16 weights in memory) is the core trick, and implementing it teaches you how memory bandwidth drives inference optimization.

---

### Week 12 — Capstone: End-to-End Kernel Engineering

**Goal:** Combine everything into a complete custom kernel project; practice the full workflow from profiling a bottleneck to shipping an optimized kernel.

**Readings**

1. FlashAttention-3 paper — Re-read with fresh eyes; you now understand every design decision.
2. Liger-Kernel (LinkedIn) or ThunderKittens (Stanford) source code — Study how production kernel libraries are organized, tested, and benchmarked.
3. Survey one frontier topic (choose one):
   - Ring Attention / Sequence Parallelism (for very long contexts)
   - Mixture-of-Experts routing kernels (Top-K gating, permute/unpermute)
   - Sparse attention patterns (block-sparse, sliding window, dilated)
   - FP8 training (Hopper/Blackwell native data types)

**Capstone Project — Profile → Identify → Optimize → Integrate**

Choose a real ML workload you care about (fine-tuning a model, running inference on a specific architecture, a custom architecture component from your research). The project follows a four-step workflow:

1. **Profile** the workload end-to-end with Nsight Systems. Identify the top-3 kernels by wall-clock time.
2. **Analyze** the bottleneck kernel with Nsight Compute: is it compute-bound or memory-bound? What is its arithmetic intensity? Where does it sit on the roofline?
3. **Implement** an optimized replacement kernel (CUDA or Triton) that targets the specific bottleneck — a fusion, a quantized variant, a tiling strategy, or an algorithmic change.
4. **Integrate** into PyTorch and validate correctness, then benchmark end-to-end speedup.

Deliverable: a 4-page technical report (IEEE or workshop format) covering motivation, profiling analysis, kernel design, and benchmarks. Include roofline plots and Nsight screenshots. Present in a 20 min + 10 min Q&A session.

---

## Reading List — Quick Reference

### Core CUDA Resources

| # | Title | Topic | Week |
|---|-------|-------|------|
| 1 | [Brief History of GPUs](https://fabiensanglard.net/cuda/) | GPU evolution | 1 |
| 2 | [GPU Glossary](https://modal.com/gpu-glossary/device-hardware/tensor-core) | Terminology reference | 1 |
| 3 | [Maximize Bandwidth & Latency (GTC)](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/) | Memory model | 1, 2 |
| 4 | [Shared Memory Microbenchmarks](https://feldmann.nyc/blog/smem-microbenchmarks) | Bank conflicts | 2 |
| 5 | [Fast Matrix Transpose on Hopper](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) | Banking & layout | 2 |
| 6 | [CUDA GEMM Step-by-Step](https://siboehm.com/articles/22/CUDA-MMM) | Tiling, optimization | 3, 5 |
| 7 | [Register-Level Tiling](https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html) | Data movement hierarchy | 3 |
| 8 | [CUDA Graphs for GenAI](https://pytorch.org/blog/accelerating-generative-ai-2/) | Launch overhead, graphs | 4 |
| 9 | [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) | Production GEMM | 5 |
| 10 | [CuTE Layouts](https://yang-yifan.github.io/blogs/cute_layout/cute_layout.html) | CUTLASS layout abstraction | 5 |
| 11 | [Tensor Core Evolution](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/) | Hardware matrix units | 6 |
| 12 | [WMMA/MMA Code](https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master/src/wmma) + [Notes](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d) | Tensor core programming | 6 |
| 13 | [Fast GEMM with Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) | From-scratch TC GEMM | 6 |
| 14 | [TMA with CuTE](https://yang-yifan.github.io/blogs/cute_tma/cute_tma.html) | Async bulk copy | 7 |
| 15 | [Blackwell GEMM (CUTLASS)](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) | Tensor Memory, 5th-gen TC | 7 |
| 16 | [Persistent Dynamic Launch](https://yang-yifan.github.io/blogs/pdl/pdl.html) | Dynamic scheduling | 7 |

### DSL Resources

| # | Title | Topic | Week |
|---|-------|-------|------|
| 17 | [Triton Lang](https://triton-lang.org/) | Python GPU DSL | 9 |
| 18 | [Modular Platform](https://www.modular.com/) | Mojo / MAX | 9 |
| 19 | [Triton vs. Modular](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls) | DSL comparison | 9 |

### ML Systems Papers

| # | Title | Topic | Week |
|---|-------|-------|------|
| 20 | FlashAttention (NeurIPS 2022) | IO-aware tiled attention | 8 |
| 21 | FlashAttention-2 (ICLR 2024) | Improved parallelism | 8 |
| 22 | FlashAttention-3 (2024) | Hopper async + FP8 | 8, 12 |
| 23 | Online softmax (Milakov & Gimelshein, 2018) | Single-pass normalizer | 3 |
| 24 | LLM.int8() (NeurIPS 2022) | INT8 inference | 11 |
| 25 | GPTQ (ICLR 2023) | Post-training quantization | 11 |
| 26 | Speculative Decoding (ICML 2023) | Parallel inference | 11 |
| 27 | PagedAttention / vLLM (SOSP 2023) | KV-cache management | 11 |
| 28 | Mixed Precision Training (Micikevicius, 2018) | FP16 training | 6 |

---

## Weekly Deliverables & Assessment

Each week, the student submits:

1. **Reading notes** (1 page): Key takeaways, one architectural insight that clicked, one question you still have.
2. **Working code**: Mini-project implementation with a README documenting build instructions, benchmarks, and a short analysis.
3. **Profiling artifacts**: At least one Nsight Systems timeline or Nsight Compute report per project demonstrating you understand the performance characteristics of your kernel.

**Grading weight** (suggested): Reading notes 20%, Mini-projects 50%, Capstone 30%.

---

## Suggested Meetings & Milestones

- **Weekly 1-on-1** (30 min): Discuss readings, review profiling results, debug kernel issues.
- **Week 4 checkpoint:** Student should be comfortable writing and profiling CUDA kernels; shared memory, reductions, and streams should feel natural. Student should be able to look at a PyTorch model and identify which operations are memory-bound vs. compute-bound.
- **Week 8 checkpoint:** Student should have implemented FlashAttention and have strong intuition for the GEMM–attention optimization hierarchy. Student should be able to read CUTLASS or FlashAttention source code and follow the logic.
- **Week 10:** Capstone proposal due (1-page description of target workload, profiling plan, and optimization hypothesis).
- **Week 12:** Capstone presentation (20 min + 10 min Q&A) and report submission.# .github
