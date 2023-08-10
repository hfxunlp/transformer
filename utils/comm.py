#encoding: utf-8

import torch.cuda.comm as comm

from utils.torch.comp import nccl_type_map

def secure_broadcast_coalesced(tensors, devices, buffer_size=10485760):

	if nccl_type_map is None:

		return comm.broadcast_coalesced(tensors, devices, buffer_size=buffer_size)
	else:
		src_type = [para.dtype for para in tensors]
		map_type = [nccl_type_map[para.dtype] if para.dtype in nccl_type_map else None for para in tensors]
		rs = comm.broadcast_coalesced([para if typ is None else para.to(typ, non_blocking=True) for para, typ in zip(tensors, map_type)], devices, buffer_size=buffer_size)

		return list(zip(*[para if mtyp is None else [pu.to(styp, non_blocking=True) for pu in para] for para, mtyp, styp in zip(list(zip(*rs)), map_type, src_type)]))
