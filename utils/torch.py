#encoding: utf-8

def bmv(inputm, inputv):

	return inputm.bmm(inputv.unsqueeze(-1)).squeeze(-1)
