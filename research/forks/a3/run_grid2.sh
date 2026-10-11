#!/bin/bash
# A3 stage 1: 2-lens grid (specs from make_specs.py grid2), raw tables, the instruction set's
# own WRITE (bqsm_vm.c apply_ramp, width N, +1 only).
# STRICT=0: this search ran before fork_lens gained strict settle; strict tables are a subset of
# these, so the raw search is the superset the strict re-check filters (RESULTS_run03.md §3).
cd "$(dirname "$0")/.."
rm -f a3/grid2/raw_vm16_?.csv
for k in 0 1 2 3 4 5; do
  ( STRICT=0 LABEL=grid2_raw_vm16 ./fork_lens list a3/grid2/specs_$k.txt a3/grid2/raw_vm16_$k.csv > a3/grid2/raw_vm16_$k.log ) &
done
wait; cat a3/grid2/raw_vm16_?.csv > a3/grid2/raw_vm16_all.csv; echo GRID2_DONE
