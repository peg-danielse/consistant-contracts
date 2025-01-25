#!/bin/bash

for i in {1..3}; do
    echo "Run chineese $i"
    python -m torch.distributed.run experiment.py "$CHECKPOINT_DIR" 0.6 0.9 15000 4 -1 "./results" 1
done
