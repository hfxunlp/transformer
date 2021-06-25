# parallel

## `base.py`

Implementations of `DataParallelModel` and `DataParallelCriterion` which support effective multi-GPU training and evaluation.

## `parallelMT.py`

Implementation of `DataParallelMT` which supports parallel decoding over multiple GPUs.

## `optm.py`

Implementation of `MultiGPUOptimizer` which performs optimization steps in parallel across multiple GPUs and `MultiGPUGradScaler`.
