#!/usr/bin/env bash
# Boot-to-pregame profiled benchmark. $1=label, $2=exe (default target/release).
set -u
cd "C:/Users/Administrator/Desktop/RBMK Project/Dream64"
LABEL="${1:?need label}"
EXE="${2:-./target/release/dream64-server.exe}"
LOG="scratchpad/bench-${LABEL}.log"
mkdir -p scratchpad
: > "$LOG"
export DREAM64_BOOT_MAX_SLICES=1
export DREAM64_PROFILE_INSTRUCTIONS=1
export DREAM64_PROFILE_PROC_STEPS=1
unset DREAM64_ACTIVATION_MAX_SLICES DREAM64_HEAP_IDENTITY_CEILING DREAM64_SCHEDULER_WALL_BUDGET_MS DREAM64_DISABLE_READY_CACHE DREAM64_PROFILE_NUMERIC_BLOCKS
START=$(date +%s)
echo "bench-${LABEL} start=$(date -Is) head=$(git rev-parse --short HEAD) exe=${EXE}" | tee -a "$LOG"
"$EXE" boot "../Monkestation2.0/tgstation.d64" >> "$LOG" 2>&1
RC=$?; END=$(date +%s)
echo "bench-${LABEL} end=$(date -Is) rc=$RC wall_s=$((END-START)) wall_min=$(( (END-START)/60 ))" | tee -a "$LOG"
echo "=== markers ==="
grep -E "startup=accepting|lobby=pregame|did not enter pregame|activation complete slices" "$LOG"
echo "=== field quickening ==="
grep -E "field_quickening hits=" "$LOG" | tail -1
