#!/bin/bash

SCRIPT=scripts/train_mnist_logreg_sparsity.py

# for num_to_remove in 2 10 100 500; do
#   for remove_type in "random" "tail" "tail_pos" "tail_neg"; do
#     PYTHONPATH=$PYTHONPATH:. python $SCRIPT $num_to_remove $remove_type
#   done
# done

PYTHONPATH=$PYTHONPATH:. python $SCRIPT 500 tail
for remove_type in "tail_pos" "tail_neg"; do
  for num_to_remove in 100 500; do
    PYTHONPATH=$PYTHONPATH:. python $SCRIPT $num_to_remove $remove_type
  done
done

PYTHONPATH=$PYTHONPATH:. python $SCRIPT 1000 random
PYTHONPATH=$PYTHONPATH:. python $SCRIPT 2000 random
PYTHONPATH=$PYTHONPATH:. python $SCRIPT 3000 random
PYTHONPATH=$PYTHONPATH:. python $SCRIPT 4000 random
PYTHONPATH=$PYTHONPATH:. python $SCRIPT 5000 random
