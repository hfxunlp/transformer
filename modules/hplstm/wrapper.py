#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear
from modules.rnncells import ATRCell, LSTMCell4RNMT
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class LSTM4RNMT(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		super(LSTM4RNMT, self).__init__()

		_osize = parse_none(osize, isize)
		self.net = LSTMCell4RNMT(isize, osize=_osize, dropout=dropout)

		self.init_cx = nn.Parameter(torch.zeros(1, _osize))
		self.init_hx = nn.Parameter(torch.zeros(1, _osize))

	def forward(self, inpute, states=None, head_mask=None, **kwargs):

		if states is None:
			bsize = inpute.size(0)
			_state = (self.init_hx.expand(bsize, -1), self.init_cx.expand(bsize, -1),)
			out = []
			for tmp in inpute.unbind(1):
				_state = self.net(tmp, _state)
				out.append(_state[0])
			out = torch.stack(out, dim=1)
		else:
			if states == "init":
				bsize = inpute.size(0)
				_state = (self.init_hx.expand(bsize, -1), self.init_cx.expand(bsize, -1),)
			else:
				_state = states
			states_return = self.net(inpute.select(1, -1), _state)
			out = states_return[0].unsqueeze(1)

		if states is None:
			return out
		else:
			return out, states_return

	def fix_init(self):

		with torch_no_grad():
			self.init_cx.zero_()
			self.init_hx.zero_()

class ATR(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		super(ATR, self).__init__()

		self.net = ATRCell(isize)

		self.init_hx = nn.Parameter(torch.zeros(1, isize))

	def forward(self, inpute, states=None, head_mask=None, **kwargs):

		if states is None:
			bsize = inpute.size(0)
			_state = self.init_hx.expand(bsize, -1)
			out = []
			for tmp in inpute.unbind(1):
				_state = self.net(tmp, _state)
				out.append(_state)
			out = torch.stack(out, dim=1)
		else:
			if states == "init":
				bsize = inpute.size(0)
				_state = self.init_hx.expand(bsize, -1)
			else:
				_state = states
			_out = self.net(inpute.select(1, -1), _state)
			states_return = _out
			out = _out.unsqueeze(1)

		if states is None:
			return out
		else:
			return out, states_return

	def fix_init(self):

		with torch_no_grad():
			self.init_hx.zero_()

class RNN(ATR):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_osize = parse_none(osize, isize)
		_hsize = _osize * 4 if hsize is None else hsize

		super(RNN, self).__init__(isize, num_head=num_head, osize=_osize, fhsize=_hsize, dropout=dropout)

		self.net = nn.Sequential(Linear(isize + _osize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))
		if dropout > 0.0:
			self.net.insert(3, Dropout(dropout, inplace=inplace_after_Custom_Act))

	def forward(self, inpute, states=None, head_mask=None, **kwargs):

		if states is None:
			bsize = inpute.size(0)
			_state = self.init_hx.expand(bsize, -1)
			out = []
			for tmp in inpute.unbind(1):
				_state = self.net(torch.cat((tmp, _state,), dim=-1))
				out.append(_state)
			out = torch.stack(out, dim=1)
		else:
			if states == "init":
				bsize = inpute.size(0)
				_state = self.init_hx.expand(bsize, -1)
			else:
				_state = states
			_out = self.net(torch.cat((inpute.select(1, -1), _state,), dim=-1))
			states_return = _out
			out = _out.unsqueeze(1)

		if states is None:
			return out
		else:
			return out, states_return
