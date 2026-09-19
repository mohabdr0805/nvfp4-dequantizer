# nvfp4-dequantizer

Reads a NVFP4 .safetensors and outputs a F32 one. Streaming, multithreaded, on Zig 0.16

Tested on `nvidia/Llama-3.1-8B-Instruct-NVFP4` : 4.98 GB in, 28.97 GB out.

## Run

```
zig build run -- <input.safetensors> <output.safetensors> [threads] [mode]
zig build test            # decode tests
zig build decode-bench    # compare decoders bench
```

`threads` 8 by default. `mode` values : `full` (default), `read_only` / `no_write` / `write`
to test one step at a time.

## Format

4 bits : 1 for the sign, 2 for the exponent, 1 for fraction. 16 possible values : 
`±{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. 

16 values share a FP8 E4M3 scale, and the tensor a FP32 scale :

```
value = weight_scale_2 * weight_scale[block] * weight[element]
```

72 bits per block of 16; 4.5 bits per value.

Worth noting :

- Safetensors don't have a type on 4 bits, so a [4096, 4096] tensor is declared as U8 [4096, 2048].
- The first element is the low nibble, and the second the high one. Nothing in the bytes tells which one : 
  both readings give valid numbers. It got settled by correlating a dequantised line against the same model 
  before quantisation : 0.9954 correlation against 0.0384.


## Measures

20 logical processors, 32 GB RAM, NVMe.

Block decoders comparison, data kept on L2 cache, best of 7 :

| | throughput |
|---|---|
| scalar + table on compilation, vectorised by LLVM | 11.9 GB/s |
| `@Vector`, f32 bit pattern reconstructed arithmetically | 10.1 GB/s |
| `@Vector`, table read lane by lane | 9.4 GB/s |

Removing the decoding branch of the element made the compiler's auto-vectorisation possible :
from 0.76 to 12.9 GB/s. Verified on the emitted assembly : zero packed instruction to 25.


Number of threads, with 4.98 GB input :

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| read, empty cache | 2.44 s | 1.80 s | 1.61 s | 1.51 s |
| read + decode | 3.79 s | 2.65 s | 2.34 s | 2.39 s |
| full | _todo_ | | | _todo_ |

The decoding saturates at 4 out of 20 threads, which is a memory bandwidth bottleneck, not compute.

A warm system cache reads at 6 GB/s, three times the disk : all the reads above were taken after emptying it with `outils/videcache.zig`.

Sustained writes drift by ±25% over a day, with the same code and same input, from 20.6 to 33.8 s. So each measure is bracketed
by a single thread control run of the same configuration.


## Parallelism

`layout()` computes all the output offsets from the shape, before any decoding.
Then the workers share an atomic counter, and work with positional reads and writes,
without any cursor, order or locks.

**One file descriptor per worker.** When they were sharing the same one, workers 
serialised on Windows, 8 threads ran slower than one.

Trying parallelised read on Python, performance improved with more threads, so the device 
was fine, and something was wrong on my code : each worker needed its own read and write 
descriptors.


## Verification

- `zig build test` — both vectorised decoders against the auto-vectorised decode table,
  5 × 256 × 256 cases, comparing bit by bit for the `-0.0` and `0.0` special case.
- `outils/verif.py` — recomputes output from the input: both edges and every 4 MB boundary of each tensor, the exact size, and contiguous offsets.
- `reference/` — a line of `q_proj` compared with `==`, without an epsilon. Decode computes 
  multiplications only, so the same inputs in the same order give the same output bits.