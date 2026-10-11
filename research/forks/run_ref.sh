#!/bin/bash
# REF_BUILD re-run of Fork A (handoff task 1): same fork_logic.c, -DREF_BUILD (DT=0.02, libm sin, no early exit).
cd "$(dirname "$0")"
B=$PWD/fork_logic_ref
( cd out_ref/cov && /usr/bin/time -v $B > cov.log 2> cov.time ) &
( cd out_ref/rob && ROBUST=32 RSIG=0.15 /usr/bin/time -v $B > rob.log 2> rob.time ) &
( cd out_ref/spot && $B spot 0.00 400 > s_d000.log ) &
( cd out_ref/spot && $B spot 0.50 400 > s_d050.log ) &
( cd out_ref/spot && NOISE=0.15 ROBUST=32 RSIG=0.15 $B spot 0.00 400 > s_d000_noise_robust.log ) &
( cd out_ref/spot && NOISE=0.15 ROBUST=32 RSIG=0.15 $B spot 0.50 400 > s_d050_noise_robust.log ) &
( cd out_ref/spot && NOISE=0.15 $B spot 0.00 400 > s_d000_noise_raw.log ) &
( cd out_ref/spot && NOISE=0.15 $B spot 0.50 400 > s_d050_noise_raw.log ) &
wait
echo REF_ALL_DONE
