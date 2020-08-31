#encoding: utf-8

import torch
from torch.cuda.amp import autocast

from parallel.base import DataParallelModel

from utils.base import pad_tensors
from utils.fmt.base import clean_list

from threading import Lock, Thread

class DataParallelMT(DataParallelModel):

	def decode(self, *inputs, **kwargs):

		if not self.device_ids:
			return self.module.decode(*inputs, **kwargs) if self.gather_output else [self.module.decode(*inputs, **kwargs)]
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		inputs = clean_list(inputs)
		ngpu = len(inputs)
		if (len(self.device_ids) == 1) or (ngpu == 1):
			return self.module.decode(*inputs[0], **kwargs[0]) if self.gather_output else [self.module.decode(*inputs[0], **kwargs[0])]
		devices = self.device_ids[:ngpu]
		if self.nets is None:
			replicas = self.replicate(self.module, devices)
		else:
			replicas = self.nets[:ngpu]
		outputs = parallel_apply_decode(replicas, inputs, devices, kwargs)
		return self.gather(pad_tensors(outputs), self.output_device) if self.gather_output else outputs

	def train_decode(self, *inputs, **kwargs):

		if not self.device_ids:
			return self.module.train_decode(*inputs, **kwargs) if self.gather_output else [self.module.train_decode(*inputs, **kwargs)]
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		inputs = clean_list(inputs)
		ngpu = len(inputs)
		if (len(self.device_ids) == 1) or (ngpu == 1):
			return self.module.train_decode(*inputs[0], **kwargs[0]) if self.gather_output else [self.module.train_decode(*inputs[0], **kwargs[0])]
		devices = self.device_ids[:ngpu]
		if self.nets is None:
			replicas = self.replicate(self.module, devices)
		else:
			replicas = self.nets[:ngpu]
		outputs = parallel_apply_train_decode(replicas, inputs, devices, kwargs)
		return self.gather(pad_tensors(outputs), self.output_device) if self.gather_output else outputs

# update these two functions with the update of parallel_apply(https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/parallel_apply.py)

def parallel_apply_decode(modules, inputs, devices, kwargs_tup=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock()
	results = {}
	grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

	def _worker(i, module, input, kwargs, device=None):

		if not isinstance(input, (list, tuple)):
			input = (input,)
		with torch.set_grad_enabled(grad_enabled), torch.cuda.device(device), autocast(enabled=autocast_enabled):
			output = module.decode(*input, **kwargs)
		with lock:
			results[i] = output

	threads = [Thread(target=_worker, args=(i, module, input, kwargs, device)) for i, (module, input, kwargs, device) in enumerate(zip(modules, inputs, kwargs_tup, devices))]

	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	outputs = []
	for i in range(len(inputs)):
		output = results[i]
		outputs.append(output)
	return outputs

def parallel_apply_train_decode(modules, inputs, devices, kwargs_tup=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock()
	results = {}
	grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

	def _worker(i, module, input, kwargs, device=None):

		if not isinstance(input, (list, tuple)):
			input = (input,)
		with torch.set_grad_enabled(grad_enabled), torch.cuda.device(device), autocast(enabled=autocast_enabled):
			output = module.train_decode(*input, **kwargs)
		with lock:
			results[i] = output

	threads = [Thread(target=_worker, args=(i, module, input, kwargs, device)) for i, (module, input, kwargs, device) in enumerate(zip(modules, inputs, kwargs_tup, devices))]

	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	outputs = []
	for i in range(len(inputs)):
		output = results[i]
		outputs.append(output)
	return outputs
