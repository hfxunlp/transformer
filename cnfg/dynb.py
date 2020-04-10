#encoding: utf-8

from cnfg.base import *

# If the angle change is greater than or equal to the minimum value in the history * dyn_tol_alpha, perform an optimization step.
dyn_tol_alpha = 1.1
# If fails to obtain a smaller angle change after this number of steps, perform an optimization step.
dyn_tol_amin = 3

# override the maximum tokens per batch configuration in `cnfg/base.py`. If there are no less than this number of tokens in a batch, an optimization step will be performed.
tokens_optm = tokens_optm * 10

# perform optimization step only in case the angle change is smaller than update_angle.
update_angle = 90.0 / dyn_tol_alpha

# number of records of the angle change reduction.
num_dynb_his = 50

# hyper parameter for parameter sampling. Ignored in case using softmax over normalized angle change reduction (default). Uncomment corresponding lines in `utils/dynbatch.py` to enable.
select_alpha = 3.0
