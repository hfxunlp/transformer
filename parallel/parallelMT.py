#encoding: utf-8

import torch

from parallel.parallel import DataParallelModel

from utils import pad_tensors

from threading import Lock, Thread

class DataParallelMT(DataParallelModel):

	def decode(self, *inputs, **kwargs):

		if not self.device_ids:
			return [self.module(*inputs, **kwargs)]
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		if (len(self.device_ids) == 1) or (len(inputs) == 1):
			return [self.module(*inputs[0], **kwargs[0])]
		nbatch = len(inputs)
		devices = self.device_ids[:nbatch]
		if self.nets is None:
			replicas = self.replicate(self.module, devices)
		else:
			replicas = self.nets[:nbatch]
		outputs = parallel_apply_decode(replicas, inputs, devices, kwargs)
		return self.gather(pad_tensors(outputs), self.output_device) if self.gather_output else outputs

	def train_decode(self, *inputs, **kwargs):

		if not self.device_ids:
			return [self.module(*inputs, **kwargs)]
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		if (len(self.device_ids) == 1) or (len(inputs) == 1):
			return [self.module(*inputs[0], **kwargs[0])]
		nbatch = len(inputs)
		devices = self.device_ids[:nbatch]
		if self.nets is None:
			replicas = self.replicate(self.module, devices)
		else:
			replicas = self.nets[:nbatch]
		outputs = parallel_apply_train_decode(replicas, inputs, devices, kwargs)
		return self.gather(pad_tensors(outputs), self.output_device) if self.gather_output else outputs

# update these two functions with the update of parallel_apply(https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/parallel_apply.py)

def parallel_apply_decode(modules, inputs, devices, kwargs_tup=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock()
	results = {}

	def _worker(i, module, input, kwargs, device=None):

		with torch.cuda.device(device):
			# this also avoids accidental slicing of `input` if it is a Tensor
			if not isinstance(input, (list, tuple)):
				input = (input,)
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

	def _worker(i, module, input, kwargs, device=None):

		with torch.cuda.device(device):
			# this also avoids accidental slicing of `input` if it is a Tensor
			if not isinstance(input, (list, tuple)):
				input = (input,)
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
