#!/bin/bash

for num in 0.2 0.8; do
    for i in {1..3}; do
        echo "Running with temperature=$num, iteration=$i"
        python -m torch.distributed.run experiment.py "$CHECKPOINT_DIR" $num
    done
done
