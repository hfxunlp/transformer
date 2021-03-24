#encoding: utf-8

from torch import nn
from random import random
from math import ceil

from utils.base import mask_tensor_type

Dropout = nn.Dropout

class TokenDropout(Dropout):

	def __init__(self, p=0.5, inplace=False, keep_magnitude=True):

		super(TokenDropout, self).__init__(p=p, inplace=inplace)
		self.keep_magnitude = (1.0 / (1.0 - self.p)) if keep_magnitude else False

	def forward(self, inpute):

		if self.training:
			mask = inpute.new_full(inpute.size()[:-1], self.p, requires_grad=False).bernoulli().to(mask_tensor_type).unsqueeze(-1)
			out = inpute.masked_fill_(mask, 0.0) if self.inplace else inpute.masked_fill(mask, 0.0)
			if self.keep_magnitude:
				out = out * self.keep_magnitude

			return out
		else:
			return inpute

def norm(lin):

	_t = sum(lin)
	return tuple([lu / _t for lu in lin])

def sample(lin):

	_t = random()
	rs, _s = len(lin) - 1, 0.0
	for i, v in enumerate(lin):
		_s += v
		if _s >= _t:
			rs = i
			break

	return rs

class NGramDropout(Dropout):

	def __init__(self, p=0.5, inplace=False, seqdim=1, sample_p=[1.0 / tmpu for tmpu in range(1, 3 + 1)], keep_magnitude=True):

		super(NGramDropout, self).__init__(p=p, inplace=inplace)
		self.seqdim = seqdim
		self.keep_magnitude = (1.0 / (1.0 - self.p)) if keep_magnitude else False
		self.sample_p = norm([float(pu) for pu in sample_p])
		self.max_n = len(sample_p)

	def forward(self, inpute):

		if self.training:
			seql = inpute.size(self.seqdim)
			ngram = sample(self.sample_p if seql > self.max_n else norm(self.sample_p[:seql - 1])) + 1
			_msize = list(inpute.size())[:-1]
			if ngram > 1:
				nblock = ceil(float(seql) / float(ngram))
				_msize[self.seqdim] = nblock
				mask = inpute.new_full(_msize, self.p, requires_grad=False).bernoulli().to(mask_tensor_type).repeat([ngram if i == self.seqdim else 1 for i in range(len(_msize))])
				if ngram * nblock != seql:
					mask = mask.narrow(self.seqdim, 0, seql)
				mask = mask.unsqueeze(-1)
			else:
				mask = inpute.new_full(_msize, self.p, requires_grad=False).bernoulli().to(mask_tensor_type).unsqueeze(-1)
			out = inpute.masked_fill_(mask, 0.0) if self.inplace else inpute.masked_fill(mask, 0.0)
			if self.keep_magnitude:
				out = out * self.keep_magnitude

			return out
		else:
			return inpute

def reduce_model(modin):

	return reduce_model_list(modin, [Dropout, TokenDropout, NGramDropout], [lambda m: (m.p, m.inplace,), lambda m: (m.p, m.inplace, m.keep_magnitude,), lambda m: (m.p, m.inplace, m.seqdim, m.keep_magnitude, m.sample_p, m.max_n,)])
