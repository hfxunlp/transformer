#encoding: utf-8

def SampleMax(input, dim=-1, keepdim=False):

	_ics = input.cumsum(dim)
	isize = list(input.size())
	isize[dim] = 1
	_sv = input.new_empty(isize).uniform_(0.0, 1.0)

	return _ics.ge(_sv).int().cumsum(dim).eq(1).int().argmax(dim=dim, keepdim=keepdim)
