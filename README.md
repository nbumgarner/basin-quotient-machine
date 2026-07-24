# Basin Quotient Machine (BQSM)

**Integer-only neural inference on any hardware. No floats, no CUDA, no Python runtime.**

BQSM replaces floating-point matrix multiplication with integer table lookups. A 4-state (2-bit) encoding maps (activation, weight) pairs through a 16-entry transition table. On SSSE3+ CPUs, one `pshufb` instruction computes 16 multiply-accumulates simultaneously.

## Quick Start

```bash
# Build (requires cc, libc, pthread, libm)
make

# Convert a GGUF model to BQSM format
make model

# Run the server
make run
# → http://localhost:8081/
```

## API

OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, model info |
| `/v1/models` | GET | Model list |
| `/v1/chat/completions` | POST | Chat inference |

```bash
curl http://localhost:8081/health
curl -X POST http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bqsm","messages":[{"role":"user","content":"Hello"}]}'
```

## Architecture

```
Prompt → Embedding → [RMS Norm → FFN(gate·up→down) + Residual] × 26 → Output
                          ↑
                   BQSM pshufb Matmul
                   (16 MACs/instruction)
```

**Model**: Gemma 2B (26 layers, d_model=2304, FFN=9216, vocab=256000)
**Format**: Packed 2-bit BQSM v3 (623 MB on disk, 2.5 GB unpacked in RAM)
**Inference**: 1.66 GMAC forward pass, integer table-lookup only

## Performance

| Kernel | Throughput | Notes |
|--------|-----------|-------|
| pshufb SIMD (SSSE3) | 2.8 GMAC/s | 16 MACs/insn, 5.1× over scalar |
| Scalar fallback | 0.55 GMAC/s | Portable C, any CPU |
| Full 26-layer pass | 3.2s | Includes norms + quantization |

Benchmarked on: Xeon X5472 @ 3.0 GHz (SSE4.1, 2007)

## Files

| File | Description |
|------|-------------|
| `bqsm_server.c` | HTTP server with OpenAI API, web UI |
| `bqsm_model.c/h` | .bqsm model loader (v3 packed format) |
| `bqsm_kernel.h` | pshufb SIMD + scalar matmul kernels |
| `bqsm_convert_packed.py` | GGUF → BQSM v3 converter |
| `bqsm_simd.c` | Standalone pshufb kernel benchmark |
| `ring_furnace.c` | Physics engine (16-oscillator ring) |
| `bqsm_energy.py` | Landauer-bound energy analysis |
| `BQSM_ARCHITECTURE.txt` | N-ary computer blueprint |

## Requirements

- **Build**: C11 compiler, libc, pthread, libm
- **SIMD**: SSSE3+ (pshufb). Falls back to scalar on other ISAs.
- **Model**: Python 3 + `gguf` package for conversion
- **RAM**: ~3 GB for Gemma 2B (2.5 GB model + runtime)

## License

Apache 2.0 — see LICENSE

## Research

BQSM is an n-ary computer architecture based on basin-quotient state machines. The core insight: a dynamical system's attractor basins encode computation without floating-point arithmetic. Integer table lookups replace multiply-accumulate, with a 13× Landauer-bound energy advantage.

See `BQSM_ARCHITECTURE.txt` and `summary.txt` for the full research trajectory.

---

*emerging.systems*
