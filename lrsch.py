#encoding: utf-8

from torch.optim.lr_scheduler import _LRScheduler
from math import sqrt

class GoogleLR(_LRScheduler):

	def __init__(self, optimizer, dmodel, warm_steps, last_epoch=-1):
		self.cur_step = 0
		self.k = 1.0 / sqrt(dmodel)
		self.wk = 1.0 / sqrt(warm_steps) / warm_steps
		super(GoogleLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		self.cur_step += 1
		cur_lr = self.k * min(1.0 / sqrt(self.cur_step), self.cur_step * self.wk)
		return [cur_lr for i in range(len(self.base_lrs))]
