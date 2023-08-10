#encoding: utf-8

from torch.autograd import Function
from torch.utils.cpp_extension import load

try:
	import movavg_cpp
except Exception as e:
	movavg_cpp = load(name="movavg_cpp", sources=["utils/cpp/movavg.cpp"])

class MovAvgFunction(Function):

	@staticmethod
	def forward(ctx, x, dim=None, beta=0.9, inplace=False):

		out = movavg_cpp.forward(x, dim, beta, inplace)
		ctx.dim, ctx.beta = dim, beta

		return out

	@staticmethod
	def backward(ctx, grad_out):

		return movavg_cpp.backward(grad_out, ctx.dim, ctx.beta), None, None, None

MovAvgFunc = MovAvgFunction.apply
