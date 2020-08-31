#encoding: utf-8

import torch
from torch import nn
from modules.base import *
from modules.act import Custom_Act

from cnfg.ihyp import *

def prepare_initState(hx, cx, bsize):

	return hx.expand(bsize, -1), cx.expand(bsize, -1)

# per gate layer normalization is applied in this implementation

# actually FastLSTM
class LSTMCell4RNMT(nn.Module):

	# isize: input size of Feed-forward NN
	# dropout: dropout over hidden units, disabling it and applying dropout to outputs (_out) in most cases

	def __init__(self, isize, osize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default):

		super(LSTMCell4RNMT, self).__init__()

		_osize = isize if osize is None else osize

		# layer normalization is also applied for the computation of hidden for efficiency. bias might be disabled in case provided by LayerNorm
		self.trans = Linear(isize + _osize, _osize * 4, bias=enable_bias)
		self.normer = nn.LayerNorm((4, _osize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.act = Custom_Act() if custom_act else nn.Tanh()
		self.drop = Dropout(dropout, inplace=inplace_after_Custom_Act) if dropout > 0.0 else None

		self.osize = _osize

	def forward(self, inpute, state):

		_out, _cell = state

		osize = list(_out.size())
		osize.insert(-1, 4)

		_comb = self.normer(self.trans(torch.cat((inpute, _out,), -1)).view(osize))

		(ig, fg, og,), hidden = _comb.narrow(-2, 0, 3).sigmoid().unbind(-2), self.act(_comb.select(-2, 3))

		if self.drop is not None:
			hidden = self.drop(hidden)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell

class GRUCell4RNMT(nn.Module):

	# isize: input size of Feed-forward NN

	def __init__(self, isize, osize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default):

		super(GRUCell4RNMT, self).__init__()

		_osize = isize if osize is None else osize

		self.trans = Linear(isize + _osize, _osize * 2, bias=enable_bias)
		self.transi = Linear(isize, _osize)
		self.transh = Linear(_osize, _osize)

		self.normer = nn.LayerNorm((2, _osize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.act = Custom_Act() if custom_act else nn.Tanh()
		self.drop = Dropout(dropout, inplace=inplace_after_Custom_Act) if dropout > 0.0 else None

		self.osize = _osize

	def forward(self, inpute, state):

		osize = list(state.size())
		osize.insert(-1, 2)

		_comb = self.normer(self.trans(torch.cat((inpute, state,), -1)).view(osize)).sigmoid()

		ig, fg = _comb.unbind(-2)

		hidden = self.act(self.transi(inpute) + ig * self.transh(state))

		if self.drop is not None:
			hidden = self.drop(hidden)

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
