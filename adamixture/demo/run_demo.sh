#!/bin/bash
set -u

echo "Running ADAMIXTURE on demo data..."
start=$(date +%s)

ADAMIXTURE_CMD=${ADAMIXTURE_CMD:-adamixture}

echo "Using command: ${ADAMIXTURE_CMD}"
command -v "${ADAMIXTURE_CMD}" 2>/dev/null || true

if "${ADAMIXTURE_CMD}" --help 2>&1 | grep -q -- "--algorithm"; then
  BRQN_ARGS="--algorithm brqn --init svd"
else
  BRQN_ARGS="--original"
fi

"${ADAMIXTURE_CMD}" \
  --k 7 \
  --data_path data/demo_data.bed \
  --save_dir outputs \
  --name demo_run \
  --device cpu \
  --seed 42 \
  --threads 1 \
  $BRQN_ARGS

status=$?
end=$(date +%s)
runtime=$((end - start))

if [ "$status" -ne 0 ]; then
  echo "Demo failed after ${runtime} seconds."
  exit "$status"
fi

echo "Demo run in ${runtime} seconds."
echo "Running diagnostics..."
"${PYTHON:-python}" run_diagnostics.py
