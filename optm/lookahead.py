#encoding: utf-8

import torch
from torch.optim.optimizer import Optimizer

class Lookahead(Optimizer):

	def __init__(self, params, optimizer, steps=5, alpha=0.8, pullback_momentum=None, **kwargs):

		super(Lookahead, self).__init__(params, {})

		self.optimizer = optimizer
		self.cur_step = 0
		self.alpha = alpha
		self.steps = steps
		self.pullback_momentum = pullback_momentum.lower()

	def step(self, closure=None):

		loss = self.optimizer.step(closure)
		self.cur_step += 1

		if self.cur_step >= self.steps:
			self.cur_step = 0
			# Lookahead and cache the current optimizer parameters
			for group in self.optimizer.param_groups:

				for p in group["params"]:

					if p.grad is not None:

						state = self.state[p]

						if len(state) == 0:
							state["cached_params"] = p.data.clone()
							if self.pullback_momentum == "pullback":
								state["cached_mom"] = p.data.new_zeros(p.data.size())
						else:
							p.data.mul_(self.alpha).add_(1.0 - self.alpha, state["cached_params"])
							state["cached_params"].copy_(p.data)
							if self.pullback_momentum == "pullback":
								internal_momentum = self.optimizer.state[p]["momentum_buffer"]
								self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
									1.0 - self.alpha, state["cached_mom"])
								state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
							elif self.pullback_momentum == "reset":
								self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

		return loss
