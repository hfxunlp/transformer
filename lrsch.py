#encoding: utf-8

from torch.optim.lr_scheduler import _LRScheduler
from math import sqrt

class GoogleLR(_LRScheduler):

	def __init__(self, optimizer, dmodel, warm_steps, scale=1.0, last_epoch=-1):

		self.cur_step = 0
		self.k = 1.0 / sqrt(dmodel)
		self.wk = 1.0 / sqrt(warm_steps) / warm_steps
		self.scale = scale
		super(GoogleLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = self.k * min(1.0 / sqrt(self.cur_step), self.cur_step * self.wk)
		if self.scale != 1.0:
			cur_lr *= self.scale
		return [cur_lr for i in range(len(self.base_lrs))]

class ReverseSqrtLR(_LRScheduler):

	def __init__(self, optimizer, lr=1e-4, scalar=1.0, min_lr=None, last_epoch=-1):

		self.cur_step = 0
		self.base_lr = lr
		self.epoch_steps = scalar
		self.min_lr = (lr / 512.0) if min_lr is None else min_lr
		super(ReverseSqrtLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):

		self.cur_step += 1
		cur_lr = max(min(1.0, 1.0 / sqrt(self.cur_step / self.epoch_steps)), self.min_lr) * self.base_lr
		return [cur_lr for i in range(len(self.base_lrs))]
