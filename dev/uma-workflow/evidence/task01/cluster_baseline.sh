#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.env"

audit_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/frust-uma-task01-XXXXXX")
trap 'rm -rf "$audit_dir"' EXIT
cd "$audit_dir"

cat > water.xyz <<'EOF'
3
Water baseline for OET UMA energy and gradient
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
printf 'oet_executable=%s\n' "$OET_TOOLS/bin/oet_uma"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$OET_TOOLS/bin/oet_uma" -t omol -m uma-s-1p1 -d cpu -o True water.ext > baseline.stdout 2>&1
cat baseline.stdout
cat water.engrad
