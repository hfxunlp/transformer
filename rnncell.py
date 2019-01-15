#encoding: utf-8

import torch
from torch import nn
from torch.nn import functional as nnFunc
from modules import *

# per gate layer normalization is applied in this implementation

# actually FastLSTM
class LSTMCell(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, osize, norm_pergate=False, use_GeLU=False):

		super(LSTMCell, self).__init__()

		# layer normalization is also applied for the computation of hidden for efficiency
		self.trans = nn.Sequential(nn.LayerNorm(isize + osize, eps=1e-06), nn.Linear(isize + osize, osize * 4)) if norm_pergate else nn.Linear(isize + osize, osize * 4)

		self.act = nn.Tanh() if use_GeLU else nn.Tanh()

		self.osize = osize

	def forward(self, inpute, state):

		_out, _cell = state

		_comb = self.trans(torch.cat((inpute, _out), -1))

		_combg, hidden = _comb.narrow(-1, 0, self.osize * 3).sigmoid(), self.act(_comb.narrow(-1, self.osize * 3, self.osize))

		ig, fg, og = _combg.narrow(-1, 0, self.osize), _combg.narrow(-1, self.osize, self.osize), _combg.narrow(-1, self.osize + self.osize, self.osize)

		_cell = fg * _cell + ig * hidden
		_out = og * self.act(_cell)

		return _out, _cell

class GRUCell(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, osize, norm_pergate=False, use_GeLU=False):

		super(GRUCell, self).__init__()

		self.trans = nn.Sequential(nn.LayerNorm(isize + osize, eps=1e-06), nn.Linear(isize + osize, osize * 2)) if norm_pergate else nn.Linear(isize + osize, osize * 2)
		self.transi = nn.Linear(isize, osize)
		self.transh = nn.Linear(osize, osize)

		self.act = nn.Tanh() if use_GeLU else nn.Tanh()

		self.osize = osize

	def forward(self, inpute, state):

		_comb = self.trans(torch.cat((inpute, state), -1)).sigmoid()

		ig, fg = _comb.narrow(-1, 0, self.osize), _comb.narrow(-1, self.osize, self.osize)

		hidden = self.act(self.transi(inpute) + ig * self.transh(state))
		_out = (1.0 - fg) * hidden + fg * state

		return _out

# ATR from: Simplifying Neural Machine Translation with Addition-Subtraction Twin-Gated Recurrent Networks
class ATRCell(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, norm_pergate=False):

		super(ATRCell, self).__init__()

		self.t1 = nn.Sequential(nn.LayerNorm(isize, eps=1e-06), nn.Linear(isize, isize)) if norm_pergate else nn.Linear(isize, isize)
		self.t2 = nn.Sequential(nn.LayerNorm(isize, eps=1e-06), nn.Linear(isize, isize)) if norm_pergate else nn.Linear(isize, isize)

	# x: input to the cell
	# cell: cell to update

	def forward(self, x, cell):

		p, q = self.t1(x), self.t2(cell)

		igate, fgate = torch.sigmoid(p + q), torch.sigmoid(p - q)

		return igate * p + fgate * q
