#encoding: utf-8

from torch import nn

from modules.base import ResidueCombiner
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.base import align_modules_by_type
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, num_sub=1, comb_input=True, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__()

		self.nets = nn.ModuleList([EncoderLayerBase(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_sub)])

		self.combiner = ResidueCombiner(isize, num_sub + 1 if comb_input else num_sub, _fhsize)

		self.comb_input = comb_input

	def forward(self, inputs, mask=None, **kwargs):

		out = inputs
		outs = [out] if self.comb_input else []
		for net in self.nets:
			out = net(out, mask)
			outs.append(out)

		return self.combiner(*outs)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=False, num_sub=1, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, num_sub, i != 0) for i in range(num_layer)])

	def load_base(self, base_encoder):

		self.drop = base_encoder.drop

		self.wemb = base_encoder.wemb

		self.pemb = base_encoder.pemb

		self.nets = align_modules_by_type(base_encoder.nets, EncoderLayerBase, self.nets)# base_encoder.nets if isinstance(base_encoder, Encoder) else

		self.out_normer = None if self.out_normer is None else base_encoder.out_normer
		self.nets[-1].combiner.out_normer = base_encoder.out_normer
