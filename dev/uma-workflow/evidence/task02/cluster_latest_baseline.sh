#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.env"
runtime=/lustre/hpc/kemi/jmni/software/oet-uma-2p23-cpu
audit_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/frust-uma-task02-XXXXXX")
trap 'rm -rf "$audit_dir"' EXIT
cd "$audit_dir"

cat > water.xyz <<'EOF'
3
Water baseline for UMA plus GFN2-xTB ALPB(chloroform)
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
EOF
cat > water.ext <<'EOF'
water.xyz
0
1
1
1
EOF

printf 'host=%s\njob=%s\nnode_list=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
printf 'runtime=%s\nxtb=%s\n' "$runtime" "$XTB_EXE"
if ! env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$runtime/bin/oet_uma" -t omol -m uma-s-1p2p1 -d cpu -o True \
  --inference-settings batch --xtb-alpb chloroform --xtb-exe "$XTB_EXE" \
  water.ext > oet.stdout 2>&1; then
  cat oet.stdout
  exit 1
fi
cat oet.stdout
cat water.engrad
cat water.uma.json
