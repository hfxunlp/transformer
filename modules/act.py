#encoding: utf-8

import torch
from torch import nn

from math import sqrt

from cnfg.ihyp import *

# 2 kinds of GELU activation function implementation according to https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L53-L58

class GeLU_GPT(nn.Module):

	def __init__(self):

		super(GeLU_GPT, self).__init__()

		self.k = sqrt(2.0 / pi)

	def forward(self, x):

		return 0.5 * x * (1.0 + (self.k * (x + 0.044715 * x.pow(3.0))).tanh())

class GeLU_BERT(nn.Module):

	def __init__(self):

		super(GeLU_BERT, self).__init__()

		self.k = sqrt(2.0)

	def forward(self, x):

		return 0.5 * x * (1.0 + (x / self.k).erf())

# Swish approximates GeLU when beta=1.702 (https://mp.weixin.qq.com/s/LEPalstOc15CX6fuqMRJ8Q).
# GELU is nonmonotonic function that has a shape similar to Swish with beta = 1.4 (https://arxiv.org/abs/1710.05941).
class Swish(nn.Module):

	def __init__(self, beta=1.0, freeze_beta=True, isize=None):

		super(Swish, self).__init__()

		if freeze_beta:
			self.beta = None if beta == 1.0 else beta
			self.reset_beta = None
		else:
			self.reset_beta = beta
			self.beta = nn.Parameter(torch.tensor([beta])) if isize is None else nn.Parameter(torch.tensor([beta]).repeat(isize))

	def forward(self, x):

		return (x.sigmoid() * x) if self.beta is None else (x * (self.beta * x).sigmoid())

	def fix_init(self):

		if self.reset_beta is not None:
			self.beta.fill_(self.reset_beta)

if override_GeLU_Swish:
	GeLU = Swish
elif override_GeLU_Sigmoid:
	GeLU = nn.Sigmoid
else:
	GeLU = GeLU_BERT
