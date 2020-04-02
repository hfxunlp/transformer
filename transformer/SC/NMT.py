#encoding: utf-8

from torch import nn

from utils.fmt.base import parse_double_value_tuple

from transformer.SC.Encoder import Encoder
from transformer.SC.Decoder import Decoder

from math import sqrt

from cnfg.ihyp import *

class NMT(nn.Module):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=False, forbidden_index=None):

		super(NMT, self).__init__()

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, num_head, xseql, ahsize, norm_output, num_layer)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

	def forward(self, inpute, inputo, mask=None):

		_mask = inpute.eq(0).unsqueeze(1) if mask is None else mask

		return self.dec(*self.enc(inpute, _mask), inputo, _mask)

	# inpute: source sentences from encoder (bsize, seql)
	# beam_size: the beam size for beam search
	# max_len: maximum length to generate

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0):

		mask = inpute.eq(0).unsqueeze(1)

		_max_len = inpute.size(1) + max(64, inpute.size(1) // 4) if max_len is None else max_len

		return self.dec.decode(*self.enc(inpute, mask), mask, beam_size, _max_len, length_penalty)
