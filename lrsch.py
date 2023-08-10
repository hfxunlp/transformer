#encoding: utf-8

from math import sqrt
from torch.optim.lr_scheduler import _LRScheduler

from cnfg.ihyp import init_lr

class GoogleLR(_LRScheduler):

	def __init__(self, optimizer, dmodel, warm_steps, scale=1.0, last_epoch=-1, **kwargs):

		self.cur_step, self.warm_steps = 0, warm_steps
		self.k = scale / sqrt(dmodel)
		self.wk = self.k / (warm_steps ** 1.5)
		super(GoogleLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = (self.cur_step * self.wk) if self.cur_step < self.warm_steps else (self.k / sqrt(self.cur_step))

		return [cur_lr for i in range(len(self.base_lrs))]

# inverse square root with warm up, portal from: https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py, equal to GoogleLR when warm_end_lr = 1.0 / sqrt(dmodel * warm_steps)
class WarmUpInverseSqrtLR(_LRScheduler):

	def __init__(self, optimizer, warm_end_lr, warm_steps, warm_init_lr=0.0, last_epoch=-1, **kwargs):

		self.cur_step, self.warm_end_lr, self.warm_steps, self.warm_init_lr = 0, warm_end_lr, warm_steps, warm_init_lr
		self.lr_step = (warm_end_lr - warm_init_lr) / warm_steps
		self.decay_factor = warm_end_lr * sqrt(warm_steps)

		super(WarmUpInverseSqrtLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = (self.warm_init_lr + self.cur_step * self.lr_step) if self.cur_step < self.warm_steps else (self.decay_factor / sqrt(self.cur_step))

		return [cur_lr for i in range(len(self.base_lrs))]

"""
class GoogleLR(WarmUpInverseSqrtLR):

	def __init__(self, optimizer, dmodel, warm_steps, scale=1.0, last_epoch=-1, **kwargs):

		super(GoogleLR, self).__init__(optimizer, scale / sqrt(dmodel * warm_steps), warm_steps, warm_init_lr=0.0, last_epoch=last_epoch)"""

class InverseSqrtLR(_LRScheduler):

	def __init__(self, optimizer, lr=1e-4, scalar=1.0, min_lr=None, last_epoch=-1, **kwargs):

		self.cur_step = 0
		self.base_lr = lr
		self.epoch_steps = scalar
		self.min_lr = (lr / 512.0) if min_lr is None else min_lr
		super(InverseSqrtLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = max(min(1.0, 1.0 / sqrt(self.cur_step / self.epoch_steps)), self.min_lr) * self.base_lr

		return [cur_lr for i in range(len(self.base_lrs))]

class CustLR(_LRScheduler):

	def __init__(self, optimizer, lr_func=lambda a, b: (init_lr, b,), ctx=None, last_epoch=-1, **kwargs):

		self.cur_step = 0
		self.lr_func = lr_func
		self.ctx = ctx
		super(CustLR, self).__init__(optimizer, last_epoch=last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr, self.ctx = self.lr_func(self.cur_step, self.ctx)

		return [cur_lr for i in range(len(self.base_lrs))]
