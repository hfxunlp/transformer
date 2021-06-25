#encoding: utf-8

from torch.autograd import Function

try:
	import lgate_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	lgate_cpp = load(name="lgate_cpp", sources=["modules/hplstm/cpp/lgate.cpp"])

class LGateFunction(Function):

	@staticmethod
	def forward(ctx, fgate, igh, init_cell, dim=None, inplace=False):

		cell = lgate_cpp.forward(fgate, igh, init_cell, dim, inplace)
		ctx.save_for_backward(cell, fgate, init_cell)
		ctx.dim = dim

		return cell

	@staticmethod
	def backward(ctx, grad_cell):

		cell, fgate, init_cell = ctx.saved_variables
		grad_fgate, grad_igh, grad_init_cell = lgate_cpp.backward(grad_cell, cell, fgate, init_cell, ctx.dim)
		return grad_fgate, grad_igh, grad_init_cell, None, None

LGateFunc = LGateFunction.apply
