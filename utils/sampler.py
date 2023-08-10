#encoding: utf-8

from utils.torch.ext import multinomial

def SampleMax(x, dim=-1, keepdim=False):

	out = multinomial(x, 1, replacement=True, dim=dim)

	return out if keepdim else out.squeeze(dim)
