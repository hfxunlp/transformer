#encoding: utf-8

# Portal from: https://github.com/juntang-zhuang/Adabelief-Optimizer

import torch
from torch.optim.optimizer import Optimizer

from math import sqrt

class AdaBelief(Optimizer):

	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):

		defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
		super(AdaBelief, self).__init__(params, defaults)

		self.weight_decouple, self.rectify, self.fixed_decay = weight_decouple, rectify, fixed_decay

	@torch.no_grad()
	def step(self, closure=None):

		if closure is None:
			loss = None
		else:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:

			for p in group['params']:

				if p.grad is not None:

					grad = p.grad
					amsgrad = group['amsgrad']

					state = self.state[p]

					beta1, beta2 = group['betas']

					if len(state) == 0:
						state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
						state['step'] = 0
						state['exp_avg'] = p.data.new_zeros(p.data.size())
						state['exp_avg_sq'] = p.data.new_zeros(p.data.size())
						if amsgrad:
							state['max_exp_avg_sq'] = p.data.new_zeros(p.data.size())

					exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

					state['step'] += 1
					bias_correction1 = 1.0 - beta1 ** state['step']
					bias_correction2 = 1.0 - beta2 ** state['step']

					if self.weight_decouple:
						if self.fixed_decay:
							p.data.mul_(1.0 - group['weight_decay'])
						else:
							p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
					elif group['weight_decay'] != 0:
						grad.add_(p.data, alpha=group['weight_decay'])

					exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
					grad_residual = grad - exp_avg
					exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)

					if amsgrad:
						max_exp_avg_sq = state['max_exp_avg_sq']
						torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

						denom = (max_exp_avg_sq.add_(group['eps']).sqrt() / sqrt(bias_correction2)).add_(group['eps'])
					else:
						denom = (exp_avg_sq.add_(group['eps']).sqrt() / sqrt(bias_correction2)).add_(group['eps'])

					if self.rectify:
						state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (1.0 - beta2 ** state['step'])

						if state['rho_t'] > 4:
							rho_inf, rho_t = state['rho_inf'], state['rho_t']
							rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t
							rt = sqrt(rt)
							step_size = rt * group['lr'] / bias_correction1
							p.data.addcdiv_(exp_avg, denom, value=-step_size)
						else:
							p.data.add_(exp_avg, alpha=-group['lr'])
					else:
						step_size = group['lr'] / bias_correction1
						p.data.addcdiv_(exp_avg, denom, value=-step_size)

		return loss
