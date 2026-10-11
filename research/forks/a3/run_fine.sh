#!/bin/bash
# A3 stage 1b: 2-lens 0.05 grid around the dipole (specs from make_specs.py fine), raw tables.
# STRICT=0 for the same reason as run_grid2.sh.
cd "$(dirname "$0")/.."
rm -f a3/fine/raw_vm16_?.csv
for k in 0 1 2 3 4 5; do ( STRICT=0 LABEL=fine_raw_vm16 ./fork_lens list a3/fine/specs_$k.txt a3/fine/raw_vm16_$k.csv > a3/fine/raw_vm16_$k.log ) & done
wait; cat a3/fine/raw_vm16_?.csv > a3/fine/raw_vm16_all.csv; echo FINE_DONE
