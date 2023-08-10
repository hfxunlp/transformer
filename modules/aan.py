#encoding: utf-8

import torch
from torch import nn

from modules.base import Custom_Act, Dropout, Linear
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

# Average Attention is proposed in Accelerating Neural Transformer via an Average Attention Network (https://aclanthology.org/P18-1166/)

class AverageAttn(nn.Module):

	# isize: input size of Feed-forward NN
	# hsize: hidden size of Feed-forward NN
	# dropout: dropout rate for Feed-forward NN
	# enable_ffn: using FFN to process the average bag-of-words representation
	# num_pos: maximum length of sentence cached, extended length will be generated while needed and droped immediately after that

	def __init__(self, isize, hsize=None, dropout=0.0, enable_ffn=False, num_pos=cache_len_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(AverageAttn, self).__init__()

		_hsize = parse_none(hsize, isize)

		self.num_pos = num_pos
		self.register_buffer("w", torch.Tensor(num_pos, 1), persistent=False)

		if enable_ffn:
			self.ffn = nn.Sequential(Linear(isize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize, bias=enable_proj_bias), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(Linear(isize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_proj_bias))
		else:
			self.ffn = None

		_d_isize = isize + isize
		self.gate = Linear(_d_isize, _d_isize)

		self.reset_parameters()

	# iQ: keys (bsize, seql, vsize) for training, (bsize, 1, vsize) for decoding
	# iV: values (bsize, seql, vsize)
	# decoding: training state or decoding state

	def forward(self, iQ, iV, decoding=False, **kwargs):

		if decoding:
			avg = iV
		else:
			seql = iV.size(1)

			# avg: (bsize, seql, vsize)
			avg = iV.cumsum(dim=1) * (self.get_ext(seql) if seql > self.num_pos else self.w.narrow(0, 0, seql))

		if self.ffn is not None:
			avg = self.ffn(avg)

		igate, fgate = self.gate(torch.cat((iQ, avg), -1)).sigmoid().chunk(2, -1)

		return igate * iQ + fgate * avg

	def reset_parameters(self):

		self.w = self.get_ext(self.num_pos)

	def get_ext(self, npos):

		return (torch.arange(1, npos + 1, dtype=self.w.dtype, device=self.w.device).reciprocal_()).unsqueeze(-1)
