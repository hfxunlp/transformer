#encoding: utf-8

import torch
import torch.cuda.comm as comm
from collections import OrderedDict
from threading import Lock, Thread
from torch import nn
#from torch._C import ScriptMethod
from torch.jit import ScriptModule
from torch.nn import DataParallel

from parallel.optm import MultiGPUOptimizer
from utils.base import divide_para_ind, filter_para_grad, filter_para_grad_iter, range_parameter_iter, reorder_by_sort
from utils.comm import secure_broadcast_coalesced
from utils.contpara import get_all_contiguous_parameters_m, get_contiguous_parameters_m, get_contiguous_parameters_p
from utils.fmt.base import clean_list
from utils.torch.comp import torch_autocast, torch_inference_mode, torch_is_autocast_enabled, torch_is_grad_enabled, torch_is_inference_mode_enabled, torch_no_grad, torch_set_grad_enabled, using_inference_mode

"""	Example:

		>>> net = DataParallelModel(model, device_ids=[0, 1, 2])
		>>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
		>>> y = net(x)
		>>> loss = criterion(y, target)
"""

def replicate_fixing(module):

	if hasattr(module, "c_available") and hasattr(module, "c_build_cache") and module.c_available():
		module.c_build_cache()

class DataParallelModel(DataParallel):

	# host replicates should improve a little bit performance if there are additional calls to update_replicas and collect_gradients in the training scripts.
	def __init__(self, module, device_ids=None, output_device=None, dim=0, host_replicate=False, gather_output=True, **kwargs):

		super(DataParallelModel, self).__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

		self.is_contiguous_parameters = False
		if host_replicate and self.device_ids and (len(self.device_ids) > 1):
			self.make_replicas()
		else:
			self.nets = None
		self.gather_output = gather_output
		self.lock = Lock()
		self.optm_splt = None
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
			outputs = parallel_apply(replicas, inputs, devices, kwargs, lock=self.lock)
			if self.gather_output:
				return self.gather(outputs, self.output_device)
			else:
				return tuple(zip(*outputs)) if isinstance(outputs[0], tuple) else outputs

	def zero_grad(self, set_to_none=True):

		if self.is_contiguous_parameters:
			with torch_no_grad():
				for para in get_all_contiguous_parameters_m(self.module):
					para.grad.zero_()
				if self.nets is not None and self.ngradev > 1:
					for net in self.nets[1:self.ngradev]:
						for para in get_all_contiguous_parameters_m(net):
							para.grad.zero_()
		else:
			for para in filter_para_grad(self.module.parameters()):
				para.grad = None
			if self.nets is not None and self.ngradev > 1:
				for net in self.nets[1:self.ngradev]:
					for para in filter_para_grad(net.parameters()):
						para.grad = None
		self.ngradev = 0

	# below 2 functions support direct access to the wrapped module parameters/modules, but exclude direct access to copies (self.nets)
	def named_parameters(self, prefix="", recurse=True):

		return self.module.named_parameters(prefix=prefix, recurse=recurse)

	def named_modules(self, memo=None, prefix="", remove_duplicate=True):

		return self.module.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)

	def make_replicas(self):

		self.nets = nn.ModuleList(replicate(self.module, self.device_ids, True))
		for net in self.nets[1:]:
			net.apply(replicate_fixing)
		self.ngradev = 0

	def collect_gradients(self):

		if self.optm_splt is not None:
			if self.is_contiguous_parameters:
				for i, (net, device,) in enumerate(zip(self.nets, self.device_ids)):
					_dev_grads = [[para.grad for para in get_contiguous_parameters_m(_net, index=i)] for _net in self.nets[:self.ngradev]]
					if i > 0:
						_dev_grads.insert(0, _dev_grads.pop(i) if i < self.ngradev else [para.grad for para in get_contiguous_parameters_m(net, index=i)])
					_dev_grads = comm.reduce_add_coalesced(_dev_grads, device)
					for mp, grad in zip(get_contiguous_parameters_m(net, index=i), _dev_grads):
						mp.grad.copy_(grad)
			else:
				grads = [[para.grad for para in filter_para_grad(net.parameters())] for net in self.nets[:self.ngradev]]
				for i, (net, device, (lind, rind,),) in enumerate(zip(self.nets, self.device_ids, self.optm_splt)):
					_dev_grads = [gradu[lind:rind] for gradu in grads]
					if i > 0:
						_dev_grads.insert(0, _dev_grads.pop(i) if i < self.ngradev else [_pg.new_zeros(_pg.size(), device=device) for _pg in _dev_grads[0]])
					_dev_grads = comm.reduce_add_coalesced(_dev_grads, device)
					for mp, grad in zip(range_parameter_iter(net, lind, rind, func=filter_para_grad_iter), _dev_grads):
						mp.grad = grad
		elif self.ngradev > 1:
			if self.is_contiguous_parameters:
				grads = comm.reduce_add_coalesced([[para.grad for para in get_all_contiguous_parameters_m(net)] for net in self.nets[:self.ngradev]], self.output_device)
				for mp, grad in zip(get_all_contiguous_parameters_m(self.module), grads):
					mp.grad.copy_(grad)
			else:
				# in case some parameters might not be used during the forward propagation on some GPUs: p.data.new_zeros(p.data.size()) if p.grad is None else p.grad instead of p.grad, but in most cases, this can warn you in case you miss the use of some parameters in the forward computation.
				grads = comm.reduce_add_coalesced([[para.grad for para in filter_para_grad(net.parameters())] for net in self.nets[:self.ngradev]], self.output_device)# if self.ngradev > 1 else [p.grad for p in filter_para_grad(self.nets[0].parameters())]
				for mp, grad in zip(filter_para_grad(self.module.parameters()), grads):
					mp.grad = grad

# the parallelization of the update of parameters can be supported, but not adviced, since the cost of multi threads is much higher and thus slower than the loop unless you are running on lots of GPUs.
# Note that gradients will be cleared every time this function was called
	def update_replicas(self):

		if self.optm_splt is None:
			if self.is_contiguous_parameters:
				params = [para.data for para in get_all_contiguous_parameters_m(self.module)]
				param_copies = comm.broadcast_coalesced(params, self.device_ids)
				with torch_no_grad():
					for module, param_copy in zip(self.nets[1:], param_copies[1:]):
						for mp, para in zip(get_all_contiguous_parameters_m(module), param_copy):
							mp.data.copy_(para)
							mp.grad.zero_()
			else:
				params = [para.data for para in filter_para_grad(self.module.parameters())]
				param_copies = comm.broadcast_coalesced(params, self.device_ids)
				# currently, pytorch broadcast binds parameters between self.nets[0] and self.module, so the following line ensures correctness but less efficient
				#for module, param_copy in zip(self.nets, param_copies):
				for module, param_copy in zip(self.nets[1:], param_copies[1:]):
					for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
						mp.data, mp.grad = para, None
		else:
			for i, (net, (lind, rind,),) in enumerate(zip(self.nets, self.optm_splt)):
				_dev_params = [para.data for para in get_contiguous_parameters_m(net, index=i)] if self.is_contiguous_parameters else [para.data for para in range_parameter_iter(net, lind, rind, func=filter_para_grad_iter)]
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
			if self.is_contiguous_parameters:
				with torch_no_grad():
					for module, param_copy in zip(self.nets, param_copies):
						for mp, para in zip(get_all_contiguous_parameters_m(module), param_copy):
							mp.data.copy_(para)
							mp.grad.zero_()
			else:
				for module, param_copy in zip(self.nets, param_copies):
					for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
						mp.data, mp.grad = para, None

		self.ngradev = 0

	def update_replicas_para(self):

		if self.optm_splt is None:
			if self.is_contiguous_parameters:
				params = [para.data for para in get_all_contiguous_parameters_m(self.module)]
				param_copies = comm.broadcast_coalesced(params, self.device_ids)
				with torch_no_grad():
					for module, param_copy in zip(self.nets[1:], param_copies[1:]):
						for mp, para in zip(get_all_contiguous_parameters_m(module), param_copy):
							mp.data.copy_(para)
			else:
				params = [para.data for para in filter_para_grad(self.module.parameters())]
				param_copies = comm.broadcast_coalesced(params, self.device_ids)
				for module, param_copy in zip(self.nets[1:], param_copies[1:]):
					for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
						mp.data = para
		else:
			for i, (net, (lind, rind,),) in enumerate(zip(self.nets, self.optm_splt)):
				_dev_params = [para.data for para in get_contiguous_parameters_m(net, index=i)] if self.is_contiguous_parameters else [para.data for para in range_parameter_iter(net, lind, rind, func=filter_para_grad_iter)]
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
				if self.is_contiguous_parameters:
					with torch_no_grad():
						for module, param_copy in zip(self.nets, param_copies):
							for mp, para in zip(get_all_contiguous_parameters_m(module), param_copy):
								mp.data.copy_(para)
				else:
					for module, param_copy in zip(self.nets, param_copies):
						for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
							mp.data = para

		self.ngradev = 0

	def collect_gradients_func(self, func):

		if self.ngradev > 1:
			grads = comm.reduce_add_coalesced([[p.grad for p in filter_para_grad(func(net).parameters())] for net in self.nets[:self.ngradev]], self.output_device)
			if self.is_contiguous_parameters:
				for mp, grad in zip(filter_para_grad(func(self.module).parameters()), grads):
					mp.grad.copy_(grad)
			else:
				for mp, grad in zip(filter_para_grad(func(self.module).parameters()), grads):
					mp.grad = grad

	def zero_replicas_grad(self, func=None):

		if self.nets is not None and self.ngradev > 1:
			if func is None:
				if self.is_contiguous_parameters:
					for net in self.nets[1:self.ngradev]:
						for para in get_all_contiguous_parameters_m(net):
							para.grad.zero_()
				else:
					for net in self.nets[1:self.ngradev]:
						for para in filter_para_grad(net.parameters()):
							para.grad = None
			else:
				if self.is_contiguous_parameters:
					for net in self.nets[1:self.ngradev]:
						for para in filter_para_grad(func(net).parameters()):
							para.grad.zero_()
				else:
					for net in self.nets[1:self.ngradev]:
						for para in filter_para_grad(func(net).parameters()):
							para.grad = None

	def build_optimizer(self, optm_func, *optm_args, multi_gpu_optimizer=False, contiguous_parameters=False, **optm_kwargs):

		self.is_contiguous_parameters = contiguous_parameters
		paras = filter_para_grad(self.module.parameters())
		if (not multi_gpu_optimizer) or self.nets is None or (len(paras) < 2):
			if contiguous_parameters:
				if self.nets is not None:
					for net in self.nets[1:]:
						get_contiguous_parameters_m(net)
				_mp = get_contiguous_parameters_m(self.module)
			else:
				_mp = self.module.parameters()
			return optm_func(_mp, *optm_args, **optm_kwargs)
		else:
			self.optm_splt, _np = divide_para_ind(paras, len(self.device_ids), return_np=True)
			if contiguous_parameters:
				for net in self.nets:
					get_contiguous_parameters_p([list(range_parameter_iter(net, lind, rind, func=filter_para_grad_iter)) for lind, rind in self.optm_splt], model=net)
				optml = [optm_func(get_contiguous_parameters_m(net, index=i), *optm_args, **optm_kwargs) for i, net in enumerate(self.nets)]
			else:
				optml = [optm_func(range_parameter_iter(net, lind, rind, func=filter_para_grad_iter), *optm_args, **optm_kwargs) for net, (lind, rind,) in zip(self.nets, self.optm_splt)]
			# sort the optimizers with slightly more parameters ahead to start their optimization steps earlier
			optml, _device_ids = reorder_by_sort(_np, optml, self.device_ids[:len(optml)], reverse=True)
			return MultiGPUOptimizer(optml, device_ids=_device_ids)

class DataParallelCriterion(DataParallel):

	# if there is no parameter update in criterion, turn on replicate_once should improve a little bit performance.
	def __init__(self, module, device_ids=None, output_device=None, dim=0, replicate_once=False, **kwargs):

		super(DataParallelCriterion, self).__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

		if replicate_once and self.device_ids and (len(self.device_ids) > 1):
			self.nets = nn.ModuleList(replicate(self.module, self.device_ids, True))
		else:
			self.nets = None
		self.lock = Lock()

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
		outputs = criterion_parallel_apply(replicas, inputs, targets, devices, kwargs, lock=self.lock)

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
		if buf.requires_grad and no_gradient:
			buffers_rg.append(clear_gradient(buf))
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
				# we have to initialize ScriptModule properly so that it works with pybind11
				replica = module._replicate_for_data_parallel()
				replica._former_parameters = OrderedDict()
				"""replica = ScriptModule()

				attribute_names = set(entry[0] for entry in module._c._get_attributes())

				keys = set(module.__dict__.keys()) - scriptmodule_skip_attr - attribute_names
				for key in keys:
					if not isinstance(module.__dict__[key], ScriptMethod):
						replica.__dict__[key] = module.__dict__[key]
				for name, the_type, value in module._c._get_attributes():
					if not name in module._buffers.keys():
						replica._c._register_attribute(name, the_type, value)"""
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

# update below functions with the update of parallel_apply(https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/parallel_apply.py)

def parallel_apply_inference(modules, inputs, devices, kwargs_tup=None, lock=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock() if lock is None else lock
	results = {}
	grad_enabled, autocast_enabled, inference_mode_enabled = torch_is_grad_enabled(), torch_is_autocast_enabled(), torch_is_inference_mode_enabled()

	def _worker(i, module, input, kwargs, device=None):

		# this also avoids accidental slicing of `input` if it is a Tensor
		if not isinstance(input, (list, tuple,)):
			input = (input,)
		with torch_set_grad_enabled(grad_enabled), torch_inference_mode(inference_mode_enabled), torch.cuda.device(device), torch_autocast(enabled=autocast_enabled):
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

def criterion_parallel_apply_inference(modules, inputs, targets, devices, kwargs_tup=None, lock=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock() if lock is None else lock
	results = {}
	grad_enabled, autocast_enabled, inference_mode_enabled = torch_is_grad_enabled(), torch_is_autocast_enabled(), torch_is_inference_mode_enabled()

	def _worker(i, module, input, target, kwargs, device):

		if not isinstance(input, (list, tuple,)):
			input = (input,)
		if not isinstance(target, (list, tuple,)):
			target = (target,)
		with torch_set_grad_enabled(grad_enabled), torch_inference_mode(inference_mode_enabled), torch.cuda.device(device), torch_autocast(enabled=autocast_enabled):
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

def parallel_apply_grad(modules, inputs, devices, kwargs_tup=None, lock=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock() if lock is None else lock
	results = {}
	grad_enabled, autocast_enabled = torch_is_grad_enabled(), torch_is_autocast_enabled()

	def _worker(i, module, input, kwargs, device=None):

		if not isinstance(input, (list, tuple,)):
			input = (input,)
		with torch_set_grad_enabled(grad_enabled), torch.cuda.device(device), torch_autocast(enabled=autocast_enabled):
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

def criterion_parallel_apply_grad(modules, inputs, targets, devices, kwargs_tup=None, lock=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock() if lock is None else lock
	results = {}
	grad_enabled, autocast_enabled = torch_is_grad_enabled(), torch_is_autocast_enabled()

	def _worker(i, module, input, target, kwargs, device):

		if not isinstance(input, (list, tuple,)):
			input = (input,)
		if not isinstance(target, (list, tuple,)):
			target = (target,)
		with torch_set_grad_enabled(grad_enabled), torch.cuda.device(device), torch_autocast(enabled=autocast_enabled):
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

parallel_apply, criterion_parallel_apply = (parallel_apply_inference, criterion_parallel_apply_inference,) if using_inference_mode else (parallel_apply_grad, criterion_parallel_apply_grad,)
