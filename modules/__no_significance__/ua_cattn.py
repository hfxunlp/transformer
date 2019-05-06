#encoding: utf-8

from math import sqrt

from torch import nn

from modules.base import SparseNormer, MHSparseNormer

class CrossAttn(nn.Module):

	# isize: input dimension
	# hsize: hidden dimension
	# osize: output size of this layer
	# num_head: number of heads
	# dropout: dropout probability
	# sparsenorm: using sparse normer or standard softmax

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=False, sparsenorm=False):

		super(CrossAttn, self).__init__()

		self.attn_dim = hsize // num_head
		self.hsize = self.attn_dim * num_head
		self.num_head = num_head

		self.query_adaptor = nn.Linear(isize, self.hsize, bias=enable_bias)
		self.kv_adaptor = nn.Linear(isize, self.hsize * 2, bias=enable_bias)

		self.outer = nn.Linear(self.hsize, osize, bias=enable_bias)

		#self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
		self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

		self.drop = nn.Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

	# iQ: query (bsize, num_query, vsize)
	# iK: keys (bsize, seql, vsize)
	# mask (bsize, num_query, seql)

	def forward(self, iQ, iK, mask=None):

		bsize, nquery, _ = iQ.size()
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		# real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
		# real_iK: MultiHead iK (bsize, seql, vsize) => (bsize, nheads, seql, adim)
		# real_iV: MultiHead iV (bsize, seql, vsize) => (bsize, nheads, seql, adim)

		real_iQ, _out = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2), self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim)

		real_iK, real_iV = _out.unbind(2)
		real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		# scores (bsize, nheads, nquery, adim) * (bsize, nheads, adim, seql) => (bsize, nheads, nquery, seql)

		scores = real_iQ.matmul(real_iK) / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1).expand_as(scores), -1e32)

		_rscore = scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		# oMA: output of MultiHeadAttention T((bsize, nheads, nquery, seql) * (bsize, nheads, seql, adim)) => (bsize, nquery, nheads, adim)

		oMA = scores.matmul(real_iV).transpose(1, 2).contiguous()

		# output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

		return self.outer(oMA.view(bsize, nquery, self.hsize)), _rscore
