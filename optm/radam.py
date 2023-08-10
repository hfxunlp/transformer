#encoding: utf-8

# Portal from: https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py

import torch
from math import sqrt
from torch.optim.optimizer import Optimizer

from utils.torch.comp import torch_no_grad

class RAdam(Optimizer):

	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, N_sma_threshhold=5, degenerated_to_sgd=True, **kwargs):

		defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
		super(RAdam, self).__init__(params, defaults)

		self.N_sma_threshhold = N_sma_threshhold
		self.degenerated_to_sgd = degenerated_to_sgd

	@torch_no_grad()
	def step(self, closure=None):

		if closure is None:
			loss = None
		else:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:

			for p in group["params"]:

				if p.grad is not None:

					state = self.state[p]

					if len(state) == 0:
						state["step"] = 0
						state["exp_avg"] = p.data.new_zeros(p.data.size())
						state["exp_avg_sq"] = p.data.new_zeros(p.data.size())

					exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
					beta1, beta2 = group["betas"]

					exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, p.grad, p.grad)
					exp_avg.mul_(beta1).add_(1 - beta1, p.grad)

					_cur_step = state["step"] = state["step"] + 1
					buffered = group["buffer"][int(_cur_step % 10)]
					if _cur_step == buffered[0]:
						N_sma, step_size = buffered[1], buffered[2]
					else:
						buffered[0] = _cur_step
						beta2_t = beta2 ** _cur_step
						N_sma_max = 2 / (1 - beta2) - 1
						N_sma = N_sma_max - 2 * _cur_step * beta2_t / (1 - beta2_t)
						buffered[1] = N_sma

						# more conservative since it"s an approximated value
						if N_sma >= self.N_sma_threshhold:
							step_size = sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** _cur_step)
						elif self.degenerated_to_sgd:
							step_size = 1.0 / (1 - beta1 ** _cur_step)
						else:
							step_size = -1
						buffered[2] = step_size

					if group["weight_decay"] > 0.0:
						p.data.add_(-group["weight_decay"] * group["lr"], p.data)

					# more conservative since it"s an approximated value
					if N_sma >= self.N_sma_threshhold:
						denom = exp_avg_sq.sqrt().add_(group["eps"])
						p.data.addcdiv_(-step_size * group["lr"], exp_avg, denom)
					elif step_size > 0:
						p.data.add_(-step_size * group["lr"], exp_avg)

		return loss
