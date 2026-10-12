# int8 × int8 dot-product kernel (aarch64)

The kernel and measurements behind the research note
[An int8 dot-product kernel takes the engine to 5.3 tokens per second on four ARM cores, output unchanged](https://emerging.systems/research/2026-10-11-arm-int8-kernel.html).

The engine it plugs into is [`nbumgarner/bqsm-neuron`](https://github.com/nbumgarner/bqsm-neuron). There the
weights are int8 with one scale per row, and each matrix product widens them to float32 against float32
activations. This kernel quantises the activations too, in blocks of `QB` values with one scale per block
(absmax / 127), and multiplies int8 by int8 with `vdotq_s32`, applying the two scales once per block and per row.

| file | what it is |
|---|---|
| `int8_q8.c` | the kernel: `int8_gemm_q8(W, wscale, X, Y, nout, nin, B, qb)` |
| `q8_hook.patch` | 45 lines against `bqsm_int8.py` at bqsm-neuron `12ec130`: loads the kernel and routes the engine's products through it when `BQSM_Q8` (or `Engine.q8`) is a block size; `0` leaves the exact path untouched |
| `fid_q8.py` | agreement with the exact path: teacher-forced top-1, KL, logit cosine, greedy generations, model step time |
| `ppl_q8.py` | perplexity of real text under the exact path and the kernel |
| `pure_decode.py` | decode speed of the model loop alone, the counterpart of `llama-bench -p 0` |
| `results/` | the raw outputs the note quotes |

## Build and run

```bash
git clone https://github.com/nbumgarner/bqsm-neuron && cd bqsm-neuron
git checkout 12ec130eac34a98ba320eeec201e30a171107a25
patch -p1 < /path/to/basin-quotient-machine/research/int8-kernel/q8_hook.patch
cc -O3 -march=armv8.2-a+dotprod -fopenmp -shared -fPIC -o libint8q8.so \
   /path/to/basin-quotient-machine/research/int8-kernel/int8_q8.c
# plus the engine's own libraries and model files, as in the bqsm-neuron README
BQSM_CORPUS=wikitext103.txt:gutenberg.txt \
  python3 /path/to/basin-quotient-machine/research/int8-kernel/fid_q8.py 16 32 64 -1
BQSM_CORPUS=wikitext103.txt:gutenberg.txt \
  python3 /path/to/basin-quotient-machine/research/int8-kernel/ppl_q8.py 16 32
```

Run both scripts from the bqsm-neuron directory. `QB = -1` means one scale for the whole vector. In the
server, `BQSM_Q8=16` turns the kernel on.

## Results (Oracle Cloud Ampere A1, 4 Neoverse-N1 cores, no GPU, 2026-10-11)

| activation scaling | teacher-forced top-1 | KL (nats) | greedy, 6 prompts × 48 tokens | model step |
|---|---|---|---|---|
| float activations (exact path) | reference | — | reference | 194 ms |
| one scale per vector | 90.5% | 0.0397 | diverges after 1–37 tokens | 99 ms |
| per 64 values | 97.8% | 0.0013 | one prompt diverges at token 21 | 105 ms |
| per 32 values | 99.3% | 0.0007 | identical | 128 ms |
| **per 16 values** | **99.2%** | **0.0005** | **identical** | **110 ms** |

Perplexity over 2,048 tokens: 30.039 exact, 30.037 at `QB = 16`. End to end through the live server, with
the engine's other work included: 3.5 → 5.3 tokens per second, with identical output token ids on the
benchmark prompt (`results/speed.txt`).

## Against llama.cpp on the same machine (2026-10-12)

| engine | tok/s |
|---|---|
| llama.cpp `e937b17`, Q8_0 (about the same bytes per weight as this int8 engine) | 10.5 |
| llama.cpp `e937b17`, Q4_K_M | 13.5 |
| this engine, model loop only (`pure_decode.py 16`) | 6.9–7.0 |
| this engine as served | 5.4 |

llama.cpp is about 1.5× faster at equal precision. Raw output, commands and the excluded run:
`results/headtohead.txt`.

The kernel's cost grows almost linearly with the number of rows multiplied at once, so it speeds up a
single stream and does not make batching cheaper. Overflow bound: at most `QB/4 · 127²` per int32 lane per block.
