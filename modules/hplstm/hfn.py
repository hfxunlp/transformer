#encoding: utf-8

import torch
from torch import nn
from modules.base import Linear, Dropout
from modules.group.base import GroupLinear
from modules.act import Custom_Act
from modules.hplstm.LGate import LGateFunc

from modules.hplstm.base import HPLSTM as HPLSTMBase
from modules.hplstm.base import BiHPLSTM as BiHPLSTMBase

from cnfg.ihyp import *

class MHPLSTMCore(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default):

		super(MHPLSTMCore, self).__init__()

		_osize = isize if osize is None else osize
		_fhsize = _osize * 4 if fhsize is None else fhsize
		_head_fhsize = _fhsize // num_head
		_fhsize = _head_fhsize * num_head

		head_dim = isize // num_head
		hsize = head_dim * num_head

		self.trans_hid = nn.Sequential(GroupLinear(hsize + hsize, _fhsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, _head_fhsize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), GroupLinear(_fhsize, hsize, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(GroupLinear(hsize + hsize, _fhsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, _head_fhsize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_fhsize, hsize, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False))
		self.trans_ifg = GroupLinear(hsize + hsize, hsize + hsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False)
		self.trans_og = nn.Sequential(GroupLinear(hsize + hsize, hsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters))

		self.normer_csum = nn.LayerNorm((num_head, head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.normer_ifg = nn.LayerNorm((num_head, 2, head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.init_cx = nn.Parameter(torch.zeros(1, num_head, head_dim))

	def forward(self, heads_input, states=None, head_mask=None):

		bsize, seql, nheads, adim = heads_input.size()
		if states is None:
			csum = self.normer_csum(torch.cat((heads_input.new_zeros(bsize, 1, nheads, adim), heads_input.narrow(1, 0, seql - 1),), dim=1).cumsum(dim=1))
		else:
			_init_state = (states == "init")
			if _init_state:
				csum = self.normer_csum(heads_input.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim)
				csum_state_return = heads_input
			else:
				_csum_state = states[0]
				csum = self.normer_csum(_csum_state)
				csum_state_return = _csum_state + heads_input
		gh_input = torch.cat((heads_input, csum,), dim=-1)
		(igate, fgate,), hidden = self.normer_ifg(self.trans_ifg(gh_input).view(bsize, seql, nheads, 2, adim)).unbind(-2), self.trans_hid(gh_input)
		fgate = fgate.sigmoid()
		igh = igate.sigmoid() * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		cell = LGateFunc(fgate, igh, self.init_cx, 1, True) if states is None else igh.addcmul_(fgate, self.init_cx.unsqueeze(1) if _init_state else states[-1])
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		if states is None:
			return out
		else:
			return out, (csum_state_return, cell,)

	def fix_init(self):

		with torch.no_grad():
			self.init_cx.zero_()

class HPLSTM(HPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		_osize = isize if osize is None else osize
		_fhsize = _osize * 4 if fhsize is None else fhsize

		super(HPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, dropout=dropout, **kwargs)

		self.net = MHPLSTMCore(isize, num_head=self.num_head, osize=_osize, fhsize=_fhsize, dropout=dropout)

class BiHPLSTM(BiHPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		_osize = isize if osize is None else osize
		_fhsize = _osize * 4 if fhsize is None else fhsize

		super(BiHPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, dropout=dropout, **kwargs)

		self.net = MHPLSTMCore(isize + isize, num_head=self.num_head + self.num_head, osize=_osize + _osize, fhsize=_fhsize + _fhsize, dropout=dropout)
