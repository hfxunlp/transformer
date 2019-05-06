#encoding: utf-8

import torch
import torch.cuda.comm as comm
from torch.nn import DataParallel

from threading import Lock, Thread

"""	Example::

		>>> net = DataParallelModel(model, device_ids=[0, 1, 2])
		>>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
		>>> y = net(x)
		>>> loss = criterion(y, target)
"""

class DataParallelModel(DataParallel):

	# host replicates should improve a little bit performance if there are additional calls to update_replicas and collect_gradients in the training scripts.
	def __init__(self, module, device_ids=None, output_device=None, dim=0, host_replicate=False, gather_output = True):

		super(DataParallelModel, self).__init__(module, device_ids, output_device, dim)

		if host_replicate and self.device_ids and (len(self.device_ids) > 1):
			self.make_replicas()
		else:
			self.nets = None

		self.gather_output = gather_output

	def forward(self, *inputs, **kwargs):

		if (not self.device_ids) or (len(self.device_ids) == 1):
			return self.module(*inputs, **kwargs) if self.gather_output else [self.module(*inputs, **kwargs)]
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		if len(inputs) == 1:
			return self.module(*inputs[0], **kwargs[0]) if self.gather_output else [self.module(*inputs[0], **kwargs[0])]
		ngpu = len(inputs)
		devices = self.device_ids[:ngpu]
		replicas = self.replicate(self.module, devices) if self.nets is None else self.nets[:ngpu]
		outputs = parallel_apply(replicas, inputs, devices, kwargs)
		# uncomment following two lines if your model have multiple outputs
		#if isinstance(outputs[0], tuple):
			#outputs = tuple(zip(*outputs))
		return self.gather(outputs, self.output_device) if self.gather_output else outputs

	def train(self, mode=True):

		super(DataParallelModel, self).train(mode)

		if self.nets is not None:
			for net in self.nets:
				net.train(mode)

		return self

	def make_replicas(self):

		self.nets = replicate_model(self.module, self.device_ids, True)

	def collect_gradients(self):

		grads = comm.reduce_add_coalesced([[p.data.new_zeros(p.data.size()) if p.grad is None else p.grad for p in net.parameters()] for net in self.nets], self.output_device)
		for mp, grad in zip(self.module.parameters(), grads):
			mp.grad = grad

# the parallelization of the update of parameters is supported, but not adviced, since the cost of multi threads is much higher and thus slower than the loop unless you are running on lots of GPUs.
# Note that gradients will be cleared every time this function was called
	def update_replicas(self, parallel=False):

		params = [para.data for para in self.module.parameters()]

		if len(params) > 0:
			param_copies = tuple([t for tensors in comm.broadcast_coalesced(params, self.device_ids) for t in tensors])

			if parallel and (len(self.device_ids) > 2):
				parallel_update_parameters(self.nets[1:], [param_copies[i:i + len(params)] for i in range(len(params), len(param_copies), len(params))])
			else:
				for module, param_copy in zip(self.nets[1:], [param_copies[i:i + len(params)] for i in range(len(params), len(param_copies), len(params))]):
					for mp, para in zip(module.parameters(), param_copy):
						mp.data, mp.grad = para, None

class DataParallelCriterion(DataParallel):

	# if there is no parameter update in criterion, turn on replicate_once should improve a little bit performance.
	def __init__(self, module, device_ids=None, output_device=None, dim=0, replicate_once=False):

		super(DataParallelCriterion, self).__init__(module, device_ids, output_device, dim)

		if replicate_once and self.device_ids and (len(self.device_ids) > 1):
			self.nets = replicate_model(self.module, self.device_ids, True)
		else:
			self.nets = None

	def forward(self, inputs, *targets, **kwargs):
		# input should be already scatterd
		# scattering the targets instead
		if not self.device_ids:
			return self.module(inputs, *targets, **kwargs)
		targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
		if (len(self.device_ids) == 1) or (len(targets) == 1):
			return self.module(inputs[0], *targets[0], **kwargs[0])
		ngpu = len(inputs)
		devices = self.device_ids[:ngpu]
		replicas = self.replicate(self.module, devices) if self.nets is None else self.nets[:ngpu]
		outputs = criterion_parallel_apply(replicas, inputs, targets, devices, kwargs)

		return self.gather(outputs, self.output_device)

def parallel_update_parameters(modules, parameter_group):

	def _worker(module, parameters):

		for mp, para in zip(module.parameters(), parameters):
			mp.data, mp.grad = para, None

	threads = [Thread(target=_worker, args=(module, parameters)) for module, parameters in zip(modules, parameter_group)]

	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

# update this function with the update of replicate(https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/replicate.py)

def replicate_model(network, devices, no_gradient=False):

	def clear_gradient(para):
		if no_gradient:
			para.grad = None
		return para

	num_replicas = len(devices)

	params = [clear_gradient(para) for para in network.parameters()] if no_gradient else list(network.parameters())
	param_indices = {param: idx for idx, param in enumerate(params)}
	param_copies = tuple([t for tensors in comm.broadcast_coalesced(params, devices) for t in tensors])
	if len(params) > 0:
		param_copies = [param_copies[i:i + len(params)] for i in range(0, len(param_copies), len(params))]

	buffers = list(network.buffers())
	buffer_indices = {buf: idx for idx, buf in enumerate(buffers)}
	buffer_copies = comm.broadcast_coalesced(buffers, devices)

	modules = list(network.modules())
	module_copies = [[] for device in devices]
	module_indices = {}

	for i, module in enumerate(modules):
		module_indices[module] = i
		for j in range(num_replicas):
			replica = module.__new__(type(module))
			replica.__dict__ = module.__dict__.copy()
			replica._parameters = replica._parameters.copy()
			replica._buffers = replica._buffers.copy()
			replica._modules = replica._modules.copy()
			module_copies[j].append(replica)

	for i, module in enumerate(modules):
		for key, child in module._modules.items():
			if child is None:
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._modules[key] = None
			else:
				module_idx = module_indices[child]
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._modules[key] = module_copies[j][module_idx]
		for key, param in module._parameters.items():
			if param is None:
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._parameters[key] = None
			else:
				param_idx = param_indices[param]
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._parameters[key] = param_copies[j][param_idx].requires_grad_(param.requires_grad)
		for key, buf in module._buffers.items():
			if buf is None:
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._buffers[key] = None
			else:
				buffer_idx = buffer_indices[buf]
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._buffers[key] = buffer_copies[j][buffer_idx]

	return [module_copies[j][0] for j in range(num_replicas)]

# update these two functions with the update of parallel_apply(https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/parallel_apply.py)

def parallel_apply(modules, inputs, devices, kwargs_tup=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock()
	results = {}

	def _worker(i, module, input, kwargs, device=None):

		with torch.cuda.device(device):
			# this also avoids accidental slicing of `input` if it is a Tensor
			if not isinstance(input, (list, tuple)):
				input = (input,)
			output = module(*input, **kwargs)
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

def criterion_parallel_apply(modules, inputs, targets, devices, kwargs_tup=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock()
	results = {}

	def _worker(i, module, input, target, kwargs, device):
		with torch.cuda.device(device):
			# this also avoids accidental slicing of `input` if it is a Tensor
			if not isinstance(input, (list, tuple)):
				input = (input,)
			if not isinstance(target, (list, tuple)):
				target = (target,)
			output = module(*(input + target), **kwargs)
		with lock:
			results[i] = output

	threads = [Thread(target=_worker, args=(i, module, input, target, kwargs, device)) for i, (module, input, target, kwargs, device) in enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	outputs = []
	for i in range(len(inputs)):
		output = results[i]
		outputs.append(output)
	return outputs
