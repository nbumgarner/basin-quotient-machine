# BQSM Performance Gaps — open items, not yet resolved

## Gap 1: 2.15× standalone vs in-server (2.8 vs 1.3 GMAC/s)
- Standalone kernel: 2.8 GMAC/s at all N (2048/4096/9216)
- In-server: 1.3 GMAC/s (same kernel, same packed weights)
- Hypothesis: RMS norm + tensor lookup + residual add overhead
- Flat-N result: phase-serial accumulators = 128 bytes, L1D-resident always
- NOT explained by: accumulator L1D thrash, weight bandwidth, swap pressure

## Gap 2: Recurring exit 137 (OOM kills)
- Observed: gemma-2b-real.bqsm (623MB packed), gemma-ternary.bqsm (131MB)
- Resolved for packed weights (640MB RSS confirmed with clean build)
- Root cause: stale binaries with 2.5GB unpacked arena
- Remaining concern: multiple concurrent processes (pgrep before launch needed)

## Gap 3: Flat-N L1D test needs verification
- 2.8 GMAC/s at N=2048, 4096, 9216
- Suspicious: crosses 32KB L1D boundary without inflection
- Possible: optimizer hoisted allocation, loop sunk, N not actually varying
- Verify: print N and measured times side-by-side, check disassembly
