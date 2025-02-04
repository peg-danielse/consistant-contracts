#!/bin/bash

strings=("arabic" "chinese" "english" "german" "dutch")

# Loop over each string
for lang in "${strings[@]}"; do
	echo "using: $lang"

  for i in {1..3}; do
    python -m torch.distributed.run experiment.py "language" "$CHECKPOINT_DIR" 0.6 0.9 15000 4 -1 "./results" "$lang"
  done
done
