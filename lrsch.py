#encoding: utf-8

from torch.optim.lr_scheduler import _LRScheduler

class GoogleLR(_LRScheduler):

	def __init__(self, optimizer, dmodel, warm_steps, last_epoch=-1):
		self.cur_step = 0
		self.k = dmodel ** (-0.5)
		self.wk = warm_steps ** (-1.5)
		super(GoogleLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		self.cur_step += 1
		cur_lr = self.k * min(self.cur_step ** (-0.5), self.cur_step * self.wk)
		return [cur_lr for i in range(len(self.base_lrs))]
