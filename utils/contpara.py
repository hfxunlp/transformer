#encoding: utf-8

# WARNING: this file may create _contiguous_parameters to the model

from torch import nn

from utils.base import filter_para_grad
from utils.torch.comp import torch_no_grad

class ContiguousParams(nn.Module):

	def __init__(self, parameters=None, init_tensors=None, **kwargs):

		super(ContiguousParams, self).__init__()

		self.weights = self.pll = None

		parameters = tuple(parameters)
		if not isinstance(parameters[0], (tuple, list,)):
			parameters = (parameters,)
		if init_tensors is not None and (not isinstance(init_tensors, (tuple, list,))):
			init_tensors = (init_tensors,)

		self.allocate(parameters=parameters, init_tensors=init_tensors)
		self.bind(update=init_tensors is None)

	def allocate(self, parameters=None, init_tensors=None):

		self.pll = self.pll if parameters is None else [filter_para_grad(pl) for pl in parameters]
		cpl = []
		if init_tensors is None:
			for pl in self.pll:
				if len(pl) > 1:
					_numel = sum(para.numel() for para in pl)
					_weight = nn.Parameter(pl[0].new_empty(_numel))
					_weight.grad = pl[0].new_zeros(_numel)
					cpl.append(_weight)
				else:
					_weight = pl[0]
					if _weight.grad is None:
						_weight.grad = _weight.new_zeros(_weight.size())
					cpl.append(_weight)
		else:
			for pl, init_tensor in zip(self.pll, init_tensors):
				if len(pl) > 1:
					_numel = sum(para.numel() for para in pl) if init_tensor is None else init_tensor.numel()
					_weight = nn.Parameter(pl[0].new_empty(_numel) if init_tensor is None else init_tensor)
					_weight.grad = pl[0].new_zeros(init_tensor.numel()) if (init_tensor is None) or (init_tensor.grad is None) else init_tensor.grad
					cpl.append(_weight)
				else:
					_weight = pl[0]
					if _weight.grad is None:
						_weight.grad = _weight.new_zeros(_weight.size()) if (init_tensor is None) or (init_tensor.grad is None) else init_tensor.grad.view(_weight.size())
					cpl.append(_weight)
		self.weights = nn.ParameterList(cpl)

	def bind(self, update=True):

		with torch_no_grad():
			for pl, weight in zip(self.pll, self.weights):
				if len(pl) > 1:
					lind = 0
					for para in pl:
						rind = lind + para.numel()
						_sizes = para.size()
						if update:
							weight.data[lind:rind].copy_(para.data.view(-1))
						para.data = weight.data[lind:rind].view(_sizes)
						if update and (para.grad is not None):
							weight.grad[lind:rind].copy_(para.grad.view(-1))
						para.grad = weight.grad[lind:rind].view(_sizes)
						lind = rind

	def bind_data(self, update=True):

		with torch_no_grad():
			for pl, weight in zip(self.pll, self.weights):
				if len(pl) > 1:
					lind = 0
					for para in pl:
						rind = lind + para.numel()
						if update:
							weight.data[lind:rind].copy_(para.data.view(-1))
						para.data = weight.data[lind:rind].view(para.size())
						lind = rind

	def bind_grad(self, update=True):

		for pl, weight in zip(self.pll, self.weights):
			if len(pl) > 1:
				lind = 0
				for para in pl:
					rind = lind + para.numel()
					if update and (para.grad is not None):
						weight.grad[lind:rind].copy_(para.grad.view(-1))
					para.grad = weight.grad[lind:rind].view(para.size())
					lind = rind

def is_model_contiguous_parameters(model):

	return hasattr(model, "_contiguous_parameters")

def get_contiguous_parameters_m(model, index=0):

	if is_model_contiguous_parameters(model):
		return [model._contiguous_parameters[index]]
	else:
		_contiguous_parameters = ContiguousParams(parameters=model.parameters()).parameters()
		model._contiguous_parameters = list(_contiguous_parameters)
	return _contiguous_parameters

def get_contiguous_parameters_p(parameters, model=None):

	_contiguous_parameters = ContiguousParams(parameters=parameters).parameters()
	if model is not None:
		if is_model_contiguous_parameters(model):
			model._contiguous_parameters.extend(list(_contiguous_parameters))
		else:
			model._contiguous_parameters = list(_contiguous_parameters)

	return _contiguous_parameters

def get_all_contiguous_parameters_m(model):

	return model._contiguous_parameters

def get_model_parameters(model, contiguous_parameters=False):

	return get_contiguous_parameters_m(model) if contiguous_parameters else model.parameters()
