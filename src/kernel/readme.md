# Triton Kernel Notes

## Part 1: RMSNorm Math

### What is RMSNorm

Root Mean Square Normalization. Normalizes each row by its RMS value, then scales by a learnable weight:

```
y = (x / sqrt(mean(x^2) + eps)) * weight
```

### Step by Step

```python
# 1. Mean of squared values (not true variance — no mean subtraction)
variance = sum(x * x) / D

# 2. Reciprocal of RMS
rrms = 1.0 / sqrt(variance + eps)

# 3. Normalize
x_normed = x * rrms
```

**Why reciprocal?** `x * rrms` uses D multiplications. `x / rms` uses D divisions. Multiplication is ~4 cycles on GPU, division is ~20+ cycles. One upfront division to get the reciprocal converts D expensive divisions into D cheap multiplications.

**Why eps?** Prevents division by zero when all values are near zero. Typically 1e-6.

### Why Normalize

Matrix multiplications accumulate values: each output element is a sum of D products. Through multiple layers, magnitudes compound:

```
Layer 1 output: sum of 768 terms → ~30
Layer 2 input: sum of 768 terms of ~30 → ~900
Layer 3 input: sum of 768 terms of ~900 → ~27000
```

Without normalization, values explode (or vanish), making training unstable.

### Why Scale After Normalization

Normalization forces all dimensions into a similar range, destroying information about which dimensions are more important. The learnable weight vector restores expressiveness:

```
Before norm:  [500, -0.001, 83000]   ← chaotic, uncontrolled magnitudes
After norm:   [0.006, -0.00001, 0.998]   ← stable but flat
After scale:  [0.3, -0.001, 1.5]    ← stable AND expressive
```

The difference: before normalization, magnitudes are accidental (whatever the previous layer produced). After normalize + scale, magnitudes are intentional (explicitly learned). The scale weights stay reasonable because gradient descent penalizes large values, and the next layer's normalization resets magnitudes anyway.

### Pre-norm vs Post-norm vs QK-norm

**Pre-norm** (modern LLMs — GPT-2, LLaMA): normalize before attention/FFN

```
x → norm → attention/FFN → + residual
```

**Post-norm** (original transformer): normalize after

```
x → attention/FFN → + residual → norm
```

**QK-norm**: normalize Q and K before the dot product to prevent softmax saturation

```
scores = norm(Q) @ norm(K)^T
```

General rule: put normalization **before** any operation where magnitudes can blow up.

### float32 Upcasting

Input may be fp16/bf16 for memory savings, but normalization math (squaring, summing, square root) needs float32 precision to avoid numerical errors. The kernel upcasts on load and implicitly downcasts on store. Same behavior as PyTorch's native RMSNorm.

## Part 2: GPU Programming

### CPU vs GPU

- **CPU**: a few very fast cores, great at complex sequential tasks
- **GPU**: thousands of slow cores, great at doing the same thing to lots of data in parallel

### Core Concepts

**Kernel** — a function that runs on the GPU. Written once, executes thousands of copies in parallel.

**Program ID** — each copy knows which piece of data it handles via `tl.program_id(axis)`. The axis refers to the dimension of the launch grid:

```python
# 1D grid — one worker per row
kernel[(n_rows,)](...)
# program_id(0) → row index

# 2D grid — e.g., matrix multiply tiles
kernel[(n_rows, n_cols)](...)
# program_id(0) → row tile index
# program_id(1) → col tile index

# 3D grid
kernel[(X, Y, Z)](...)
# program_id(0), program_id(1), program_id(2)
```

**Grid** — how many kernel copies to launch. `(n_rows,)` means one kernel per row.

**Block size** — how many elements each worker processes at once. Must be a power of 2 because GPU hardware requires aligned, power-of-2 sized work. Rounded up from the actual data size using `triton.next_power_of_2(D)`.

**Mask** — when BLOCK_SIZE > D, the mask prevents reading/writing out-of-bounds memory. Masked positions get filled with a safe value (e.g., 0.0). The cores assigned to masked positions still run but do no useful work — a small tradeoff for simplicity.

**Pointers and stride** — GPU memory is flat. A 2D matrix is a 1D array row by row:

```
Matrix [[a,b,c],[d,e,f]]  →  Memory: [a, b, c, d, e, f]

row_start = X_ptr + row_idx * stride
```

`stride` = elements to skip to reach the next row. Usually equals D, but can differ for padded or non-contiguous tensors. Always use `x.stride(0)` instead of hardcoding D.

**Load → Compute → Store** — every kernel follows this pattern:

```
GPU global memory → registers (fast) → do math → registers → GPU global memory
```

### Why Kernel Fusion Matters

Without fusion, PyTorch launches a separate kernel for each op. Each kernel writes results to global memory, and the next kernel reads them back:

```
x²       → write to memory → read back
sum(x²)  → write to memory → read back
sqrt()   → write to memory → read back
...7 round trips total
```

A fused kernel does everything in one pass: one read, one write. The speedup comes from eliminating intermediate memory traffic, not from faster math.

### Memory Hierarchy

```
Registers:    ~1 cycle     (per thread, ~256 max)
L1 cache:     ~10 cycles   (per SM, shared by threads on same SM)
L2 cache:     ~30 cycles   (shared by all SMs)
Global/DRAM:  ~400 cycles  (main GPU memory)
```

### Shared Data Access (e.g., weight vector)

When multiple workers load the same data (e.g., the weight vector W), the hardware handles it efficiently:

1. Multiple SMs request the same address simultaneously
2. The memory controller deduplicates the request, fetches once from DRAM
3. Populates L2 cache, broadcasts to all requesting SMs
4. Each SM caches in L1 for its local workers

No explicit synchronization needed — the hardware is designed for this pattern.

### Register Spilling

Each SM has ~65536 registers shared across all its threads. If a kernel uses too many registers per thread, excess data **spills** to local memory (DRAM speed, ~400 cycles). Performance degrades gradually:

```
D=768   (BLOCK=1024)  → full speed, registers comfortable
D=4096  (BLOCK=4096)  → full speed
D=8192  (BLOCK=8192)  → ~80-90% speed, register pressure
D=16384 (BLOCK=16384) → ~40-60% speed, significant spilling
D=32768 (BLOCK=32768) → may not compile
```

### Tiling for Large D

Fix for large D: process the row in chunks instead of all at once. Use a fixed BLOCK_SIZE (e.g., 4096) and loop over the row:

- **Pass 1**: iterate chunks, accumulate sum of squares → compute rrms
- **Pass 2**: iterate chunks, normalize and scale → store output

Tradeoff: 2 passes over X (one extra read) vs register spilling. For large D, tiling wins because sequential reads are fast while register spills are random and slow.

### Counting Memory Passes

Look at every `tl.load` and `tl.store`. Ask: does this pointer get visited once or multiple times across the loops?

```
Single-pass RMSNorm:  X(1 read) + W(1 read) + Y(1 write) = 3 passes
Tiled RMSNorm:        X(2 reads) + W(1 read) + Y(1 write) = 4 passes
```

### Memory Bound vs Compute Bound

Calculate and compare:

```
memory_time  = total_bytes / GPU_bandwidth
compute_time = total_flops / GPU_throughput
```

Whichever is larger is the bottleneck.

**Quick shortcut — arithmetic intensity:**

```
arithmetic_intensity = flops / bytes

< 10 flops/byte   → memory bound  (normalization, activations, element-wise ops)
> 100 flops/byte  → compute bound (matmul, convolution)
```

RMSNorm is ~1 flop/byte — heavily memory bound. That's why fusion helps so much: reducing memory traffic is the only way to speed it up.

### Performance Degradation Cases

1. **D too large** — register spilling to slow local memory
2. **D far from power of 2** — e.g., D=1025 → BLOCK=2048, ~50% wasted work
3. **Too few rows** — not enough workers to keep all SMs busy
4. **Small D** — kernel launch overhead dominates actual compute
5. **Non-contiguous memory** — breaks memory coalescing, causes scattered reads
6. **Unnecessary precision cast** — `.to(tl.float32)` on already-fp32 input is wasted work

## Part 3: Benchmark Results

GPU: NVIDIA GeForce GTX 1080 Ti (12GB VRAM), dtype: bfloat16

Run: `python -m benchmark.kernel.bench_rmsnorm`

```
         D   n_rows  Triton Single-pass (us)  Triton 2D (us)  Triton Tiled (us)  Torch Builtin (us)  Torch Compile (us)  Torch Naive (us)
0    128.0     64.0                 4.096000        4.096000           4.096000            4.096000            3.712000         18.432001
1    128.0    256.0                 4.096000        4.096000           4.096000            4.096000            4.096000         20.288000
2    128.0    512.0                 5.120000        4.096000           5.120000            5.264000            4.096000         21.504000
3    128.0   1024.0                 6.144000        5.024000           7.168000            7.168000            5.120000         23.968000
4    128.0   2048.0                 9.216000        6.144000          10.240000           10.240000            7.168000         31.952001
5    128.0   4096.0                14.400000        9.216000          16.384000           17.408000           10.496000         67.552000
6    128.0   8192.0                25.599999       15.360000          28.672000           30.719999           18.271999        135.264002
7    128.0  16384.0                47.104001       27.648000          53.344000           57.344001           32.432001        252.927989
8    128.0  32768.0                90.112001       53.183999         103.423998          111.616001           61.487999        485.055998
9    128.0  65536.0               176.832005      103.423998         202.335998          219.136000          119.808003        947.200000
10   256.0     64.0                 3.712000        4.000000           4.096000            4.096000            4.096000         19.455999
11   256.0    256.0                 4.096000        4.320000           4.192000            4.784000            5.120000         20.479999
12   256.0    512.0                 5.184000        5.120000           6.032000            6.144000            5.248000         23.552001
13   256.0   1024.0                 7.168000        6.144000           7.168000            7.344000            7.792000         32.655999
14   256.0   2048.0                10.240000        8.512000          11.264000           11.264000           11.136000         66.239998
15   256.0   4096.0                17.088000       15.360000          18.352000           19.392001           17.408000        135.744005
16   256.0   8192.0                29.968000       27.648000          31.744000           34.672000           29.696001        252.927989
17   256.0  16384.0                56.320000       52.639998          59.392001           64.511999           55.296000        484.320000
18   256.0  32768.0               109.536000      103.407998         115.712002          125.952005          106.495999        945.151985
19   256.0  65536.0               214.432001      203.776002         226.303995          246.784002          208.895996       1869.824052
20   512.0     64.0                 4.096000        4.096000           4.096000            4.096000            5.120000         20.479999
21   512.0    256.0                 5.120000        5.120000           5.120000            5.120000            6.144000         23.584001
22   512.0    512.0                 7.072000        6.176000           7.168000            7.168000            7.168000         33.760000
23   512.0   1024.0                 9.216000        8.928000           9.856000           10.208000           11.392000         65.664001
24   512.0   2048.0                16.384000       15.360000          16.384000           16.384000           18.031999        136.191994
25   512.0   4096.0                28.672000       28.560000          28.816000           29.696001           30.719999        253.951997
26   512.0   8192.0                54.272000       53.247999          54.848000           55.264000           55.968001        484.351993
27   512.0  16384.0               104.447998      103.423998         105.455998          105.407998          106.495999        945.151985
28   512.0  32768.0               204.799995      203.776002         206.847996          205.824003          208.095998       1869.824052
29   512.0  65536.0               406.527996      405.503988         416.736007          412.768006          411.648005       3717.119932
30   768.0     64.0                 4.096000        4.096000           4.096000            4.960000            6.256000         21.504000
31   768.0    256.0                 5.360000        5.344000           5.552000            6.144000            7.232000         28.672000
32   768.0    512.0                 8.336000        7.456000           9.072000            9.216000            9.216000         48.864000
33   768.0   1024.0                12.288000       11.520000          12.480000           13.312000           15.360000        101.008002
34   768.0   2048.0                22.528000       21.600001          22.655999           23.552001           24.607999        194.720000
35   768.0   4096.0                41.312002       40.832002          41.983999           43.008000           44.032000        369.664013
36   768.0   8192.0                78.879997       78.624003          81.184000           82.943998           82.943998        714.752018
37   768.0  16384.0               154.624000      153.600007         159.743994          163.839996          160.991997       1407.199979
38   768.0  32768.0               305.952013      304.800004         320.159987          328.063995          324.735999       2792.095900
39   768.0  65536.0               608.255982      607.007980         644.096017          658.111989          674.816012       5563.551903
40  1024.0     64.0                 4.096000        4.352000           4.432000            5.120000            8.192000         22.528000
41  1024.0    256.0                 6.144000        6.144000           6.144000            6.144000            9.056000         32.768000
42  1024.0    512.0                10.112000        8.416000          10.240000           10.240000           10.624000         66.560000
43  1024.0   1024.0                16.303999       15.360000          16.384000           16.384000           19.455999        134.143993
44  1024.0   2048.0                28.672000       27.648000          29.696001           29.696001           32.768000        253.087997
45  1024.0   4096.0                54.272000       53.247999          57.344001           59.440000           64.511999        485.423997
46  1024.0   8192.0               104.447998      103.423998         110.592000          115.712002          121.215999        945.504010
47  1024.0  16384.0               204.799995      203.776002         221.535996          231.424004          245.759994       1868.800044
48  1024.0  32768.0               406.464010      405.503988         445.439994          460.799992          499.711990       3714.047909
49  1024.0  65536.0               808.960021      807.936013         895.904005          923.648000         1007.616043       7411.151886
50  2048.0     64.0                 5.120000        5.120000           5.120000            6.176000           12.288000         25.599999
51  2048.0    256.0                 9.216000        9.216000           9.216000            9.312000           14.336000         68.608001
52  2048.0    512.0                16.384000       16.384000          17.024000           17.152000           20.463999        134.143993
53  2048.0   1024.0                28.352000       27.648000          31.744000           32.768000           43.744002        250.880003
54  2048.0   2048.0                53.247999       53.247999          66.848002           70.656002           76.800004        483.520001
55  2048.0   4096.0               103.423998      103.423998         133.120000          141.376004          145.408005        947.135985
56  2048.0   8192.0               204.112001      203.999996         269.311994          286.720008          290.511996       1867.776036
57  2048.0  16384.0               405.503988      405.503988         541.696012          574.464023          578.559995       3709.072113
58  2048.0  32768.0               808.048010      807.936013        1095.072031         1151.999950         1150.975943       7401.472092
59  2048.0  65536.0              1613.088012     1612.800002        2184.479952         2308.095932         2308.095932      14780.655861
60  4096.0     64.0                 7.168000        6.448000           6.896000            9.216000           21.536000         36.864001
61  4096.0    256.0                15.520000       16.384000          17.408000           18.432001           26.624000        137.024000
62  4096.0    512.0                28.672000       28.672000          33.792000           36.864001           43.008000        250.959992
63  4096.0   1024.0                54.064000       53.247999          66.016003           75.776003           90.992000        482.304007
64  4096.0   2048.0               103.824001      103.840001         131.840006          149.311997          163.424000        944.127977
65  4096.0   4096.0               204.799995      204.704002         261.503994          297.984004          305.440009       1871.871948
66  4096.0   8192.0               406.143993      405.503988         522.367984          594.784021          612.352014       3711.999893
67  4096.0  16384.0               808.960021      808.144003        1043.455958         1191.536009         1207.296014       7398.399830
68  4096.0  32768.0              1613.824010     1613.824010        2085.887909         2379.776001         2404.351950      14772.736073
69  4096.0  65536.0              3224.287987     3224.048018        4175.951958         4764.159918         4798.463821      29539.327621
70  8192.0     64.0                10.240000       10.240000          10.240000           15.232000           39.264001         66.399999
71  8192.0    256.0                29.440001       28.944001          40.832002           41.983999           55.296000        253.951997
72  8192.0    512.0                54.272000       54.272000          77.823997           78.847997           84.063999        484.351993
73  8192.0   1024.0               104.447998      105.056003         151.552007          157.695994          182.991996        944.127977
74  8192.0   2048.0               205.663994      205.824003         302.080005          306.176007          328.704000       1865.728021
75  8192.0   4096.0               406.527996      406.527996         601.024002          607.231975          613.376021       3707.904100
76  8192.0   8192.0               808.960021      808.960021        1198.352039         1208.320022         1221.632004       7394.303799
77  8192.0  16384.0              1613.824010     1613.824010        2393.791914         2412.480116         2423.808098      14774.784088
78  8192.0  32768.0              3225.472093     3224.303961        4786.176205         4818.943977         4814.847946      29578.239441
79  8192.0  65536.0              6443.359852     6444.032192        9570.303917         9634.816170         9626.624107      59087.966919
```

### Key Observations

- **Triton 2D wins for small D**: at D=128, 2D is ~1.7x faster than single-pass due to row batching reducing worker scheduling overhead
- **Triton 2D matches single-pass for large D**: at D >= 2048, adaptive BLOCK_SEQ falls to 1, making it equivalent to single-pass (no register spilling)
- **All Triton kernels are 5-9x faster than Torch Naive** across all configurations
- **Triton beats Torch Builtin (F.rms_norm)**: single-pass and 2D consistently match or outperform PyTorch's built-in cuDNN-fused implementation
- **Torch Compile is competitive at small D** but falls behind at larger D and row counts
- **Speedup improves with scale**: small tensors see ~4.5x speedup, large tensors see ~9x over naive
- **Tiled kernel overhead**: the extra memory pass costs ~1.3-1.5x for D >= 2048 compared to single-pass
- **Memory bound confirmed**: throughput scales linearly with tensor size, consistent with bandwidth-limited operation

