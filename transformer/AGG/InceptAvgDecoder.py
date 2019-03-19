#encoding: utf-8

import torch
from torch import nn
from modules import *
from math import sqrt

from transformer.AvgDecoder import DecoderLayer as DecoderLayerBase
from transformer.AvgDecoder import Decoder as DecoderBase

# Average Decoder is proposed in Accelerating Neural Transformer via an Average Attention Network(https://arxiv.org/abs/1805.00631)

class DecoderLayer(nn.Module):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, num_sub=1):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__()

		self.nets = nn.ModuleList([DecoderLayerBase(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_sub)])

		self.combiner = ResidueCombiner(isize, num_sub, _fhsize)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: embedding of decoded translation (bsize, nquery, isize) during training, layer normed summed previous states for decoding
	# src_pad_mask: mask for given encoding source sentence (bsize, nquery, seql), see Encoder, expanded after generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# query_unit: single query to decode, used to support decoding for given step
	# step: current decoding step, used to average over the sum.

	def forward(self, inpute, inputo, src_pad_mask=None, query_unit=None, step=1):

		outs = []
		if query_unit is None:
			out = inputo
			states_return = None
			for net in self.nets:
				out = net(inpute, out, src_pad_mask)
				outs.append(out)
		else:
			out = query_unit
			states_return = []
			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, None if inputo is None else inputo.select(-2, _tmp), src_pad_mask, out, step)
				outs.append(out)
				states_return.append(_state)

			states_return = torch.stack(states_return, -2)

		out = self.combiner(*outs)

		if states_return is None:
			return out
		else:
			return out, states_return

class Decoder(DecoderBase):

	# construction function is needed, since DecoderLayer should be re-assigned.

	# isize: size of word embedding
	# nwd: number of words
	# num_layer: number of encoder layers
	# fhsize: number of hidden units for PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# emb_w: weight for embedding. Use only when the encoder and decoder share a same dictionary
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention
	# bindemb: bind embedding and classifier weight

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=512, ahsize=None, norm_output=False, bindemb=False, forbidden_index=None, num_sub=1):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, _fhsize, dropout, attn_drop, emb_w, num_head, xseql, _ahsize, norm_output, bindemb, forbidden_index)

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize, num_sub) for i in range(num_layer)])

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		self.pemb = base_decoder.pemb

		_nets = base_decoder.nets

		_lind = 0
		for net in self.nets:
			_rind = _lind + len(net.nets)
			net.nets = nn.ModuleList(_nets[_lind:_rind])
			_lind = _rind

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer
		self.nets[-1].combiner.out_normer = base_decoder.out_normer
