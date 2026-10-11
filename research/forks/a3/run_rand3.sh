#!/bin/bash
# A3 stage 2: 3-lens random search (specs from make_specs.py rand3, seed 20261010), raw tables,
# the instruction set's own WRITE (width N, +1). STRICT=0 as in run_grid2.sh; the 45 raw hits were
# then re-run with ROBUST=32 and STRICT=1 (default) -> rand3/strict_robust_hits.csv.
cd "$(dirname "$0")/.."
rm -f a3/rand3/raw_vm16_?.csv
for k in 0 1 2 3 4 5 6; do ( STRICT=0 LABEL=rand3_raw_vm16 ./fork_lens list a3/rand3/specs_$k.txt a3/rand3/raw_vm16_$k.csv > a3/rand3/raw_vm16_$k.log ) & done
wait; cat a3/rand3/raw_vm16_?.csv > a3/rand3/raw_vm16_all.csv; echo RAND3_DONE
