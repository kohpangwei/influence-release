#!/bin/bash

SCRIPT=scripts/train_mnist_logreg_sparsity.py

for num_to_remove in 2 10 100 500; do
  for remove_type in "random" "tail" "tail_pos" "tail_neg"; do
    PYTHONPATH=$PYTHONPATH:. python $SCRIPT $num_to_remove $remove_type
  done
done

