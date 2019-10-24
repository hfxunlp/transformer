#encoding: utf-8

import torch.cuda.comm as comm

from parallel.parallelMT import DataParallelMT as DataParallelMTBase

from utils.base import filter_para_grad

class DataParallelMT(DataParallelMTBase):

	def collect_gradients(self):

		grads = comm.reduce_add_coalesced([[p.data.new_zeros(p.data.size()) if p.grad is None else p.grad for p in filter_para_grad(net.parameters())] for net in self.nets], self.output_device)
		for mp, grad in zip(filter_para_grad(self.module.parameters()), grads):
			mp.grad = grad

	def update_replicas(self, parallel=False):

		params = [para.data for para in filter_para_grad(self.module.parameters())]

		if len(params) > 0:
			param_copies = tuple([t for tensors in comm.broadcast_coalesced(params, self.device_ids) for t in tensors])

			for module, param_copy in zip(self.nets[1:], [param_copies[i:i + len(params)] for i in range(len(params), len(param_copies), len(params))]):
				for mp, para in zip(filter_para_grad(module.parameters()), param_copy):
					mp.data, mp.grad = para, None
