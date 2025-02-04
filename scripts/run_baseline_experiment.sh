#!/bin/bash

for i in {1..3}; do
    echo "Run $i"
    python -m torch.distributed.run experiment.py "baseline" "$CHECKPOINT_DIR"
done
