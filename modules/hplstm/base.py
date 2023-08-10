#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear
from modules.group.base import GroupLinear
from modules.hplstm.LGate import LGateFunc
from utils.base import float2odd
from utils.fmt.parser import parse_none
from utils.torch.comp import flip_mask, torch_no_grad

from cnfg.ihyp import *

class MHPLSTMCore(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(MHPLSTMCore, self).__init__()

		_osize = parse_none(osize, isize)

		i_head_dim = float2odd(float(isize) / num_head)
		i_hsize = i_head_dim * num_head
		o_head_dim = float2odd(float(_osize) / num_head)
		o_hsize = o_head_dim * num_head

		self.trans_hid = GroupLinear(i_hsize + i_hsize, o_hsize * 3, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False)
		self.trans_og = nn.Sequential(GroupLinear(i_hsize + o_hsize, o_hsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, o_head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters))

		self.normer_csum = nn.LayerNorm((num_head, i_head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.normer_hid = nn.LayerNorm((num_head, 3, o_head_dim), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.act = Custom_Act() if custom_act else nn.ReLU()#Tanh()
		self.drop = Dropout(dropout, inplace=inplace_after_Custom_Act) if dropout > 0.0 else None
		self.init_cx = nn.Parameter(torch.zeros(num_head, o_head_dim))

	# heads_input: (bsize, seql, nheads, adim)
	# states: ((bsize, 1, num_head, head_dim), (bsize, 1, num_head, head_dim),)
	# head_mask: (bsize, seql, 1, 1)

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

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
		igate, fgate, hidden = self.normer_hid(self.trans_hid(torch.cat((heads_input, csum,), dim=-1)).view(bsize, seql, nheads, 3, -1)).unbind(-2)
		fgate = fgate.sigmoid()
		hidden = self.act(hidden)

		if self.drop is not None:
			hidden = self.drop(hidden)
		igh = igate.sigmoid() * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		cell = LGateFunc(fgate, igh, self.init_cx, 1, True) if states is None else igh.addcmul_(fgate, self.init_cx if _init_state else states[-1])
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		if states is None:
			return out
		else:
			return out, (csum_state_return, cell,)

	def fix_init(self):

		with torch_no_grad():
			self.init_cx.zero_()

class HPLSTM(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, dropout=0.0, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(HPLSTM, self).__init__()

		_osize = parse_none(osize, isize)
		o_hsize = float2odd(float(_osize) / num_head) * num_head

		self.head_dim = float2odd(float(isize) / num_head)
		i_hsize = self.head_dim * num_head
		self.num_head = num_head

		self.trans_input = Linear(isize, i_hsize, bias=enable_proj_bias)
		self.net = MHPLSTMCore(i_hsize, num_head=self.num_head, osize=o_hsize, dropout=dropout)
		self.trans_output = Linear(o_hsize, _osize, bias=enable_proj_bias)

	def forward(self, inpute, states=None, head_mask=None, **kwargs):

		bsize, seql = inpute.size()[:2]
		heads_input = self.trans_input(inpute).view(bsize, seql, self.num_head, self.head_dim)

		if states is None:
			out = self.net(heads_input, states=states, head_mask=head_mask)
		else:
			out, states_return = self.net(heads_input, states=states, head_mask=head_mask)

		out = self.trans_output(out.view(bsize, seql, -1))

		if states is None:
			return out
		else:
			return out, states_return

class BiHPLSTM(nn.Module):

	def __init__(self, isize, num_head=8, osize=None, dropout=0.0, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(BiHPLSTM, self).__init__()

		_osize = parse_none(osize, isize)
		o_hsize = float2odd(float(_osize) / num_head) * num_head

		self.head_dim = float2odd(float(isize) / num_head)
		i_hsize = self.head_dim * num_head
		self.num_head = num_head

		self.trans_input = Linear(isize, i_hsize + i_hsize, bias=enable_proj_bias)
		self.net = MHPLSTMCore(i_hsize + i_hsize, num_head=self.num_head + self.num_head, osize=o_hsize + o_hsize, dropout=dropout)
		self.trans_output = Linear(o_hsize + o_hsize, _osize, bias=enable_proj_bias)

	# inpute: (bsize, seql, isize)
	# mask: (bsize, seql, 1, 1), generated by input.eq(pad_id).view(bsize, seql, 1, 1)
	# pad_reversed_mask: (bsize, seql, nheads * 2, 1), generated by torch.cat((mask.new_zeros(1, 1, 1, 1).expand(bsize, seql, nheads, 1), mask.flip(1).expand(bsize, seql, nheads, 1),), dim=2)

	def forward(self, inpute, mask=None, pad_reversed_mask=None, **kwargs):

		bsize, seql = inpute.size()[:2]
		nheads = self.num_head
		heads_input_fwd, heads_input_bwd = self.trans_input(inpute).view(bsize, seql, 2, nheads, self.head_dim).unbind(2)
		heads_input_bwd_rvs = heads_input_bwd.flip(1)
		_r_mask = pad_reversed_mask if mask is None else torch.cat((mask.new_zeros(1, 1, 1, 1).expand(bsize, seql, nheads, 1), flip_mask(mask, 1).expand(bsize, seql, nheads, 1),), dim=2)
		o_fwd, o_bwd_rvs = self.net(torch.cat((heads_input_fwd, heads_input_bwd_rvs,), dim=2), head_mask=_r_mask).chunk(2, dim=-2)
		o_bwd = o_bwd_rvs.flip(1)

		return self.trans_output(torch.cat((o_fwd.view(bsize, seql, -1), o_bwd.view(bsize, seql, -1),), dim=-1))
