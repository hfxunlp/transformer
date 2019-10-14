#encoding: utf-8

import torch
from torch import nn
from modules.base import *

def prepare_initState(hx, cx, bsize):

	return hx.expand(bsize, -1), cx.expand(bsize, -1)

# per gate layer normalization is applied in this implementation

# actually FastLSTM
class LSTMCell4RNMT(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, osize, use_GeLU=False):

		super(LSTMCell4RNMT, self).__init__()

		# layer normalization is also applied for the computation of hidden for efficiency
		self.trans = Linear(isize + osize, osize * 4)
		self.normer = nn.LayerNorm((4, osize), eps=1e-06)

		self.act = GeLU_BERT() if use_GeLU else nn.Tanh()

		self.osize = osize

	def forward(self, inpute, state):

		_out, _cell = state

		_comb = self.normer(self.trans(torch.cat((inpute, _out), -1)).view(-1, 4, self.osize))

		_combg, hidden = _comb.narrow(-2, 0, 3).sigmoid(), self.act(_comb.select(-2, 3))

		ig, fg, og = _combg.unbind(-2)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell

class GRUCell4RNMT(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, osize, use_GeLU=False):

		super(GRUCell4RNMT, self).__init__()

		self.trans = Linear(isize + osize, osize * 2)
		self.transi = Linear(isize, osize)
		self.transh = Linear(osize, osize)

		self.normer = nn.LayerNorm((2, osize), eps=1e-06)

		self.act = GeLU_BERT() if use_GeLU else nn.Tanh()

		self.osize = osize

	def forward(self, inpute, state):

		_comb = self.normer(self.trans(torch.cat((inpute, state), -1)).view(-1, 2, self.osize)).sigmoid()

		ig, fg = _comb.unbind(-2)

		hidden = self.act(self.transi(inpute) + ig * self.transh(state))
		_out = (1.0 - fg) * hidden + fg * state

		return _out

# ATR from: Simplifying Neural Machine Translation with Addition-Subtraction Twin-Gated Recurrent Networks
class ATRCell(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize):

		super(ATRCell, self).__init__()

		self.t1 = Linear(isize, isize)
		self.t2 = Linear(isize, isize)

	# x: input to the cell
	# cell: cell to update

	def forward(self, x, cell):

		p, q = self.t1(x), self.t2(cell)

		igate, fgate = (p + q).sigmoid(), (p - q).sigmoid()

		return igate * p + fgate * q
