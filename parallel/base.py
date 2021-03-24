#encoding: utf-8

import torch
import torch.cuda.comm as comm
from torch.cuda.amp import autocast
from utils.comm import secure_broadcast_coalesced

from torch.jit import ScriptModule
from torch._C import ScriptMethod

from torch.nn import DataParallel

from threading import Lock, Thread

from utils.base import filter_para_grad, divide_para_ind, reorder_by_sort, range_parameter_iter, filter_para_grad_iter
from utils.fmt.base import clean_list

from parallel.optm import MultiGPUOptimizer

"""	Example::

		>>> net = DataParallelModel(model, device_ids=[0, 1, 2])
		>>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
		>>> y = net(x)
		>>> loss = criterion(y, target)
"""

class DataParallelModel(DataParallel):

	# host replicates should improve a little bit performance if there are additional calls to update_replicas and collect_gradients in the training scripts.
	def __init__(self, module, device_ids=None, output_device=None, dim=0, host_replicate=False, gather_output=True):

		super(DataParallelModel, self).__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

		if host_replicate and self.device_ids and (len(self.device_ids) > 1):
			self.make_replicas()
		else:
			self.nets = None

		self.optm_splt = None
		self.gather_output = gather_output
		self.ngradev = 0

	def forward(self, *inputs, **kwargs):

		if (not self.device_ids) or (len(self.device_ids) == 1):
			return self.module(*inputs, **kwargs) if self.gather_output else [self.module(*inputs, **kwargs)]
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		inputs = clean_list(inputs)
		ngpu = len(inputs)
		if self.training and ngpu > self.ngradev:
			self.ngradev = ngpu
		if ngpu == 1:
			outputs = self.module(*inputs[0], **kwargs[0])
			if self.gather_output:
				return outputs
			else:
				return tuple((ou,) for ou in outputs) if isinstance(outputs, tuple) else outputs
		else:
			devices = self.device_ids[:ngpu]
			replicas = self.replicate(self.module, devices) if self.nets is None else self.nets[:ngpu]
			outputs = parallel_apply(replicas, inputs, devices, kwargs)
			if self.gather_output:
				return self.gather(outputs, self.output_device)
			else:
				return tuple(zip(*outputs)) if isinstance(outputs[0], tuple) else outputs

	def train(self, mode=True):

		super(DataParallelModel, self).train(mode)

		if self.nets is not None:
			for net in self.nets[1:]:
				net.train(mode)

		return self

	def make_replicas(self):

		self.nets = replicate(self.module, self.device_ids, True)
		self.ngradev = 0

	def collect_gradients(self):

		if self.optm_splt is not None:
			grads = [[p.grad for p in filter_para_grad(net.parameters())] for net in self.nets[:self.ngradev]]
			for i, (net, device, (lind, rind,),) in enumerate(zip(self.nets, self.device_ids, self.optm_splt)):
				_dev_grads = [gradu[lind:rind] for gradu in grads]
				if i > 0:
					_dev_grads.insert(0, _dev_grads.pop(i) if i < self.ngradev else [_pg.new_zeros(_pg.size(), device=device) for _pg in _dev_grads[0]])
				_dev_grads = comm.reduce_add_coalesced(_dev_grads, device)
				for mp, grad in zip(range_parameter_iter(net, lind, rind, func=filter_para_grad_iter), _dev_grads):
					mp.grad = grad
		elif self.ngradev > 1:
			# in case some parameters might not be used during the forward propagation on some GPUs: p.data.new_zeros(p.data.size()) if p.grad is None else p.grad instead of p.grad, but in most cases, this can warn you in case you miss the use of some parameters in the forward computation.
			grads = comm.reduce_add_coalesced([[p.grad for p in filter_para_grad(net.parameters())] for net in self.nets[:self.ngradev]], self.output_device)# if self.ngradev > 1 else [p.grad for p in filter_para_grad(self.nets[0].parameters())]
			for mp, grad in zip(filter_para_grad(self.module.parameters()), grads):
				mp.grad = grad

# the parallelization of the update of parameters can be supported, but not adviced, since the cost of multi threads is much higher and thus slower than the loop unless you are running on lots of GPUs.
# Note that gradients will be cleared every time this function was called
	def update_replicas(self):

		if self.optm_splt is None:
			params = [para.data for para in filter_para_grad(self.module.parameters())]

			if len(params) > 0:
				param_copies = comm.broadcast_coalesced(params, self.device_ids)

				# currently, pytorch broadcast binds parameters between self.nets[0] and self.module, so the following line ensures correctness but less efficient
				#for module, param_copy in zip(self.nets, param_copies):
				for module, param_copy in zip(self.nets[1:], param_copies[1:]):
					for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
						mp.data, mp.grad = para, None
		else:
			for i, (net, (lind, rind,),) in enumerate(zip(self.nets, self.optm_splt)):
				_dev_params = [para.data for para in range_parameter_iter(net, lind, rind, func=filter_para_grad_iter)]
				if i > 0:
					_devices = self.device_ids[:]
					_devices.insert(0, _devices.pop(i))
				else:
					_devices = self.device_ids
				_dev_param_copies = comm.broadcast_coalesced(_dev_params, _devices)
				if i > 0:
					_dev_param_copies.insert(i, _dev_param_copies.pop(0))
					for pc, _dpc in zip(param_copies, _dev_param_copies):
						pc.extend(_dpc)
				else:
					param_copies = _dev_param_copies
			for module, param_copy in zip(self.nets, param_copies):
				for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
					mp.data, mp.grad = para, None

		self.ngradev = 0

	def update_replicas_para(self):

		if self.optm_splt is None:
			params = [para.data for para in filter_para_grad(self.module.parameters())]

			if len(params) > 0:
				param_copies = comm.broadcast_coalesced(params, self.device_ids)

				for module, param_copy in zip(self.nets[1:], param_copies[1:]):
					for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
						mp.data = para
		else:
			for i, (net, (lind, rind,),) in enumerate(zip(self.nets, self.optm_splt)):
				_dev_params = [para.data for para in range_parameter_iter(net, lind, rind, func=filter_para_grad_iter)]
				if i > 0:
					_devices = self.device_ids[:]
					_devices.insert(0, _devices.pop(i))
				else:
					_devices = self.device_ids
				_dev_param_copies = comm.broadcast_coalesced(_dev_params, _devices)
				if i > 0:
					_dev_param_copies.insert(i, _dev_param_copies.pop(0))
					for pc, _dpc in zip(param_copies, _dev_param_copies):
						pc.extend(_dpc)
				else:
					param_copies = _dev_param_copies
			for module, param_copy in zip(self.nets, param_copies):
				for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
					mp.data = para

		self.ngradev = 0

	def zero_grad(self, set_to_none=True):

		self.module.zero_grad(set_to_none=set_to_none)
		if self.nets is not None and self.ngradev > 1:
			# currently, pytorch broadcast binds parameters between self.nets[0] and self.module, so the following line ensures correctness but less efficient
			#for net in self.nets:
			for net in self.nets[1:]:
				net.zero_grad(set_to_none=set_to_none)
		self.ngradev = 0

	def collect_gradients_func(self, func):

		if self.ngradev > 1:
			grads = comm.reduce_add_coalesced([[p.grad for p in filter_para_grad(func(net).parameters())] for net in self.nets[:self.ngradev]], self.output_device)
			for mp, grad in zip(filter_para_grad(func(self.module).parameters()), grads):
				mp.grad = grad

	def zero_replicas_grad(self, func=None):

		if self.nets is not None and self.ngradev > 1:
			if func is None:
				for net in self.nets[1:self.ngradev]:
					for para in filter_para_grad(net.parameters()):
						para.grad = None
			else:
				for net in self.nets[1:self.ngradev]:
					for para in filter_para_grad(func(net).parameters()):
						para.grad = None

	def reset_grad(self):

		for para in filter_para_grad(self.module.parameters()):
			para.grad = None
		if self.nets is not None and self.ngradev > 1:
			for net in self.nets[1:self.ngradev]:
				for para in filter_para_grad(net.parameters()):
					para.grad = None
		self.ngradev = 0

	def build_optimizer(self, optm_func, *optm_args, **optm_kwargs):

		paras = filter_para_grad(self.module.parameters())
		if self.nets is None or (len(paras) < 2):
			return optm_func(self.module.parameters(), *optm_args, **optm_kwargs)
		else:
			self.optm_splt, _np = divide_para_ind(paras, len(self.device_ids), return_np=True)
			optml = [optm_func(range_parameter_iter(net, lind, rind, func=filter_para_grad_iter), *optm_args, **optm_kwargs) for net, (lind, rind,) in zip(self.nets, self.optm_splt)]
			# sort the optimizers with slightly more parameters ahead to start their optimization steps earlier
			optml, _device_ids = reorder_by_sort(_np, optml, self.device_ids[:len(optml)], reverse=True)
			return MultiGPUOptimizer(optml, device_ids=_device_ids)

class DataParallelCriterion(DataParallel):

	# if there is no parameter update in criterion, turn on replicate_once should improve a little bit performance.
	def __init__(self, module, device_ids=None, output_device=None, dim=0, replicate_once=False):

		super(DataParallelCriterion, self).__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

		if replicate_once and self.device_ids and (len(self.device_ids) > 1):
			self.nets = replicate(self.module, self.device_ids, True)
		else:
			self.nets = None

	def forward(self, inputs, *targets, **kwargs):
		# input should be already scatterd
		# scattering the targets instead
		if not self.device_ids:
			return self.module(inputs[0], *targets, **kwargs)
		targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
		targets = clean_list(targets)
		ngpu = len(targets)
		if ngpu == 1:
			return self.module(inputs[0], *targets[0], **kwargs[0])
		devices = self.device_ids[:ngpu]
		replicas = self.replicate(self.module, devices) if self.nets is None else self.nets[:ngpu]
		outputs = criterion_parallel_apply(replicas, inputs, targets, devices, kwargs)

		return self.gather(outputs, self.output_device)

# update this function with the update of replicate(https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/replicate.py)

def replicate(network, devices, no_gradient=False):

	def clear_gradient(para):
		para.grad = None
		return para

	num_replicas = len(devices) - 1

	params = [clear_gradient(para) for para in network.parameters()] if no_gradient else list(network.parameters())
	param_indices = {param: idx for idx, param in enumerate(params)}
	param_copies = comm.broadcast_coalesced(params, devices)[1:]

	buffers = list(network.buffers())
	buffers_rg = []
	buffers_not_rg = []
	for buf in buffers:
		if buf.requires_grad and not detach:
			buffers_rg.append(clear_gradient(buf) if no_gradient else buf)
		else:
			buffers_not_rg.append(buf)

	buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
	buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

	buffer_copies_rg = secure_broadcast_coalesced(buffers_rg, devices)[1:]
	buffer_copies_not_rg = secure_broadcast_coalesced(buffers_not_rg, devices)[1:]

	modules = list(network.modules())
	module_copies = [[] for device in devices]
	module_indices = {}
	scriptmodule_skip_attr = {"_parameters", "_buffers", "_modules", "forward", "_c"}

	for i, module in enumerate(modules):
		module_indices[module] = i
		for j in range(num_replicas):
			if isinstance(module, ScriptModule):
				# we have to initialize ScriptModule properly so that
				# it works with pybind11
				replica = module._replicate_for_data_parallel()
				replica._former_parameters = OrderedDict()
				'''replica = ScriptModule()

				attribute_names = set(entry[0] for entry in module._c._get_attributes())

				keys = set(module.__dict__.keys()) - scriptmodule_skip_attr - attribute_names
				for key in keys:
					if not isinstance(module.__dict__[key], ScriptMethod):
						replica.__dict__[key] = module.__dict__[key]
				for name, the_type, value in module._c._get_attributes():
					if not name in module._buffers.keys():
						replica._c._register_attribute(name, the_type, value)'''
			else:
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
					module_copies[j][i]._modules[key] = None
			else:
				module_idx = module_indices[child]
				for j in range(num_replicas):
					module_copies[j][i]._modules[key] = module_copies[j][module_idx]
		for key, param in module._parameters.items():
			if param is None:
				for j in range(num_replicas):
					module_copies[j][i]._parameters[key] = None
			else:
				param_idx, _p_require_grad = param_indices[param], param.requires_grad
				for j in range(num_replicas):
					module_copies[j][i]._parameters[key] = param_copies[j][param_idx].requires_grad_(_p_require_grad)
		for key, buf in module._buffers.items():
			if buf is None:
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._buffers[key] = None
			else:
				_p_require_grad = buf.requires_grad
				if _p_require_grad:
					buffer_copies = buffer_copies_rg
					buffer_idx = buffer_indices_rg[buf]
				else:
					buffer_copies = buffer_copies_not_rg
					buffer_idx = buffer_indices_not_rg[buf]
				for j in range(num_replicas):
					module_copies[j][i]._buffers[key] = buffer_copies[j][buffer_idx].requires_grad_(_p_require_grad)

	for j in range(num_replicas):
		for i, module in enumerate(modules):
			if isinstance(module, ScriptModule):
				replica = module_copies[j][i]
				for method_name in module._c._method_names():
					replica._c.clone_method(module._c, method_name)

	return [network] + [module_copies[j][0] for j in range(num_replicas)]

# update these two functions with the update of parallel_apply(https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/parallel_apply.py)

def parallel_apply(modules, inputs, devices, kwargs_tup=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock()
	results = {}
	grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

	def _worker(i, module, input, kwargs, device=None):

		# this also avoids accidental slicing of `input` if it is a Tensor
		if not isinstance(input, (list, tuple)):
			input = (input,)
		with torch.set_grad_enabled(grad_enabled), torch.cuda.device(device), autocast(enabled=autocast_enabled):
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
	grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

	def _worker(i, module, input, target, kwargs, device):

		if not isinstance(input, (list, tuple)):
			input = (input,)
		if not isinstance(target, (list, tuple)):
			target = (target,)
		with torch.set_grad_enabled(grad_enabled), torch.cuda.device(device), autocast(enabled=autocast_enabled):
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
