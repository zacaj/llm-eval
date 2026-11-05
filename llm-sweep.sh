#!/bin/bash
# Harbor catches Ctrl+C and exits normally, so bash would carry on to the next model;
# this trap runs once the current harbor.sh has finished (and saved its settings).
trap 'echo "sweep interrupted" >&2; exit 130' INT

# ssh -p 222 zacaj@192.168.2.15 'C:\\programs\\llm wait 120'

# DATASET=tasks/tb2_tc MAX_DUR=1200 ./harbor/harbor.sh run claude-code -i '*__tc5'

for k in 1 2 3; do

# (
cmds=(
  "strata swift-iq3_xxs"
  "strata darwin-r3-gyro-m"
  "strata gyro-m"
  "strata swift-gyro-m"
  "strata darwin-180b-rsi-r3-gsq-rco-iq3_xxs"
  # "llama darwin"
  # "llama Qwen3.8-27B-UD-IQ3_XXS-unsloth.gguf"
  # "llama Qwen3.6-35B-A3B-UD-IQ4_XS-MTP-unsloth.gguf"
  # "strata groot"
  # "strata gyro-coder-m"
  # "strata coder-iq1_m-new"
  "llama Qwen3.6-35B-A3B-UD-IQ3_XXS-MTP.gguf"
  "llama Qwen3.8-27B-IQ3_XXS-GSQ-RCO-MTP-ISNTA_DASLab.gguf"
  # "llama Qwen3.6-27B-UD-IQ3_XXS-MTP-unsloth.gguf"
)

for cmd in "${cmds[@]}"; do
  SWAP_CMD="ssh -p 222 zacaj@192.168.2.15 'C:\\programs\\llm $cmd'" \
  EFFORT=low \
  THINK_BUDGET=8000 \
  MAX_DUR_SCALE=auto \
  SKIP_EXISTING=1 \
  BUDGET=100000 \
  ./harbor/harbor.sh run claude-code \
    --list lists/qfn-diff.txt \
    -k $k
done
# )

done
