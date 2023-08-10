#encoding: utf-8

import torch
from collections import defaultdict
from threading import Thread
from torch.optim.optimizer import Optimizer

from utils.torch.comp import GradScaler#, torch_autocast, torch_inference_mode, torch_is_autocast_enabled, torch_is_grad_enabled, torch_is_inference_mode_enabled, torch_set_grad_enabled, using_inference_mode

class MultiGPUOptimizer(Optimizer):

	def __init__(self, optms, device_ids=None, **kwargs):

		torch._C._log_api_usage_once("python.optimizer")
		self.defaults = optms[0].defaults

		self._hook_for_profile()

		self.state = defaultdict(dict)
		self.param_groups = []

		for optm in optms:
			for param_group in optm.param_groups:
				self.add_param_group(param_group)

		self.optms, self.device_ids = optms, device_ids

	def step(self, closure=None):

		parallel_apply(self.optms, closure=closure, devices=self.device_ids)

	def state_dict(self):

		return [optm.state_dict() for optm in self.optms]

	def load_state_dict(self, state_dict):

		for optm, sdu in zip(self.optms, state_dict):
			optm.load_state_dict(sdu)

class MultiGPUGradScaler(GradScaler):

	def step(self, optimizer, *args, **kwargs):

		_step_func = super(MultiGPUGradScaler, self).step
		if isinstance(optimizer, MultiGPUOptimizer):
			for optm in optimizer.optms:
				_step_func(optm, *args, **kwargs)
		else:
			_step_func(optimizer, *args, **kwargs)

def parallel_apply(optms, closure=None, devices=None):

	#grad_enabled, autocast_enabled, inference_mode_enabled = torch_is_grad_enabled(), torch_is_autocast_enabled(), torch_is_inference_mode_enabled()

	def _worker(optm, closure=None, device=None):

		with torch.cuda.device(device):#, torch_set_grad_enabled(grad_enabled), torch_inference_mode(), torch_autocast(enabled=autocast_enabled)
			optm.step(closure=closure)

	threads = [Thread(target=_worker, args=(optm, closure, device)) for optm, device in zip(optms, devices)]

	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
