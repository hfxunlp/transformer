#encoding: utf-8

from torch import nn

from modules.attn.retr import ResSelfAttn
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, k_rel_pos=use_k_relative_position_encoder, smoothing=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		super(EncoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, k_rel_pos=k_rel_pos, **kwargs)

		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=self.attn.norm_residual, k_rel_pos=k_rel_pos, smoothing=smoothing, use_cumsum=False)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])
