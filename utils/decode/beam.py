#encoding: utf-8

from torch import Tensor

def repeat_bsize_for_beam_tensor(tin, beam_size):

	_tsize = list(tin.size())
	_rarg = [1 for i in range(len(_tsize))]
	_rarg[1] = beam_size
	_tsize[0] *= beam_size

	return tin.repeat(*_rarg).view(_tsize)

def expand_bsize_for_beam(*inputs, beam_size=1):

	outputs = []
	for inputu in inputs:
		if isinstance(inputu, Tensor):
			outputs.append(repeat_bsize_for_beam_tensor(inputu, beam_size))
		elif isinstance(inputu, dict):
			outputs.append({k: expand_bsize_for_beam(v, beam_size=beam_size) for k, v in inputu.items()})
		elif isinstance(inputu, tuple):
			outputs.append(tuple(expand_bsize_for_beam(tmpu, beam_size=beam_size) for tmpu in inputu))
		elif isinstance(inputu, list):
			outputs.append([expand_bsize_for_beam(tmpu, beam_size=beam_size) for tmpu in inputu])
		else:
			outputs.append(inputu)

	return outputs[0] if len(inputs) == 1 else tuple(outputs)
