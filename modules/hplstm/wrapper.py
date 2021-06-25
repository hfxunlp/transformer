#encoding: utf-8

import torch
from torch import nn

from modules.rnncells import LSTMCell4RNMT, ATRCell

class LSTM4RNMT(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		super(LSTM4RNMT, self).__init__()

		_osize = isize if osize is None else osize
		self.net = LSTMCell4RNMT(isize, osize=_osize, dropout=dropout)

		self.init_cx = nn.Parameter(torch.zeros(1, _osize))
		self.init_hx = nn.Parameter(torch.zeros(1, _osize))

	def forward(self, inpute, states=None, head_mask=None):

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

		with torch.no_grad():
			self.init_cx.zero_()
			self.init_hx.zero_()

class ATR(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		super(ATR, self).__init__()

		self.net = ATRCell(isize)

		self.init_hx = nn.Parameter(torch.zeros(1, isize))

	def forward(self, inpute, states=None, head_mask=None):

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

		with torch.no_grad():
			self.init_hx.zero_()
