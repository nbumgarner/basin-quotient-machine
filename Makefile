# BQSM Inference Server — Makefile
# Builds a zero-dependency BQSM inference server.
# Requires: cc (gcc/clang), libc, pthread, libm
# Minimum: SSE4.1 (pshufb for SIMD kernel)
# Portable: scalar fallback on non-x86

CC      := cc
CFLAGS  := -O3 -std=c11 -Wall
LDFLAGS := -lm -pthread

# Auto-detect native arch; override with MARCH= if cross-compiling
MARCH   ?= native
CFLAGS  += -march=$(MARCH)

# For old CPUs without AVX: MARCH=native works, but explicit:
#   make MARCH=core2    (SSSE3, no SSE4)
#   make MARCH=nocona   (SSE3 only, no pshufb → scalar fallback)

# Source files
SRCS    := bqsm_server.c bqsm_model.c
OBJS    := $(SRCS:.c=.o)
HEADERS := bqsm_model.h bqsm_kernel.h
BINARY  := bqsm_server

# ─── Targets ──────────────────────────────────────────────────────────────

.PHONY: all clean test run bench

all: $(BINARY)

$(BINARY): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Model conversion (requires gguf-py: pip install gguf)
model:
	python3 bqsm_convert_packed.py
	@echo "Model written to ~/Downloads/gemma-2b-full.bqsm"

# Run server with full model
run: $(BINARY)
	./$(BINARY) --model $(HOME)/Downloads/gemma-2b-full.bqsm --port 8081

# Run with demo model (3 tensors, fast load)
run-demo: $(BINARY)
	./$(BINARY) --model $(HOME)/Downloads/gemma-2b.bqsm --port 8081

# Quick test: load model, print summary, exit
test: $(BINARY)
	./bqsm_load_test $(HOME)/Downloads/gemma-2b-full.bqsm

# SIMD kernel benchmark
bench:
	$(CC) $(CFLAGS) -o bqsm_simd_bench bqsm_simd.c $(LDFLAGS)
	./bqsm_simd_bench

# Ring furnace benchmark
furnace:
	$(CC) $(CFLAGS) -o ring_furnace ring_furnace.c $(LDFLAGS)
	./ring_furnace

clean:
	rm -f $(BINARY) $(OBJS) bqsm_load_test bqsm_simd_bench ring_furnace

# ─── Install ──────────────────────────────────────────────────────────────
PREFIX ?= /usr/local
install: $(BINARY)
	install -d $(PREFIX)/bin
	install -m 755 $(BINARY) $(PREFIX)/bin/
	@echo "Installed $(BINARY) to $(PREFIX)/bin/"
