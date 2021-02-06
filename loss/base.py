#encoding: utf-8

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.modules.loss import NLLLoss as NLLLossBase

from torch.nn.functional import kl_div, nll_loss

from utils.base import clear_pad_mask, eq_indexes

from cnfg.ihyp import *

# Faster implementation from fairseq: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py#L33-L50, but do not support fbil.
def fast_label_smoothing_loss(input, target, ignore_index, conf, smoothing_value, reduction):

	_target = target.unsqueeze(-1)
	nll_loss = -input.gather(dim=-1, index=_target)
	smooth_loss = -input.sum(dim=-1, keepdim=True)
	if isinstance(ignore_index, (list, tuple)):
		pad_mask = eq_indexes(_target, ignore_index)
		nll_loss.masked_fill_(pad_mask, 0.0)
		smooth_loss.masked_fill_(pad_mask, 0.0)
	elif ignore_index >= 0:
		pad_mask = _target == ignore_index
		nll_loss.masked_fill_(pad_mask, 0.0)
		smooth_loss.masked_fill_(pad_mask, 0.0)
	if reduction != "none":
		nll_loss = nll_loss.sum()
		smooth_loss = smooth_loss.sum()
	loss = conf * nll_loss + smoothing_value * smooth_loss
	if reduction == "mean":
		loss = loss / float(target.numel())

	return loss

"""	from: Rethinking the Inception Architecture for Computer Vision (https://arxiv.org/abs/1512.00567)
	With label smoothing, KL-divergence between q_{smoothed ground truth prob.}(w) and p_{prob. computed by model}(w) is minimized.
"""

class LabelSmoothingLoss(_Loss):

	# enable fast_mode will ignore forbidden_index
	def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction='mean', forbidden_index=-1, fast_mode=use_fast_loss):

		super(LabelSmoothingLoss, self).__init__()

		self.fast_mode, self.reduction = fast_mode, reduction

		fbil = set()
		if isinstance(ignore_index, (list, tuple)):
			tmp = []
			for _tmp in ignore_index:
				if (_tmp >= 0) and (_tmp not in tmp):
					tmp.append(_tmp)
					if _tmp not in fbil:
						fbil.add(_tmp)
			_nid = len(tmp)
			if _nid > 0:
				self.ignore_index = tuple(tmp) if _nid > 1 else tmp[0]
			else:
				self.ignore_index = ignore_index[0] if len(ignore_index) > 0 else -1
		else:
			self.ignore_index = ignore_index
			if (ignore_index >= 0) and (ignore_index not in fbil):
				fbil.add(ignore_index)

		if fast_mode:
			self.smoothing_value = label_smoothing / (nclass - 1)
			self.conf = 1.0 - label_smoothing - self.smoothing_value
		else:
			if isinstance(forbidden_index, (list, tuple)):
				for fi in forbidden_index:
					if (fi >= 0) and (fi not in fbil):
						fbil.add(fi)
			else:
				if forbidden_index is not None and forbidden_index >= 0:
					fbil.add(forbidden_index)

			smoothing_value = label_smoothing / (nclass - 1 - len(fbil))

			weight = torch.full((nclass,), smoothing_value)
			weight.index_fill_(0, torch.tensor(tuple(fbil), dtype=torch.long, device=weight.device), 0.0)
			self.register_buffer("weight", weight.unsqueeze(0))
			self.conf = 1.0 - label_smoothing

	# input: (batch size, num_classes)
	# target: (batch size)
	# they will be flattened automatically if the dimension of input is larger than 2.

	def forward(self, input, target):

		if self.fast_mode:

			return fast_label_smoothing_loss(input, target, self.ignore_index, self.conf, self.smoothing_value, self.reduction)
		else:
			_input = input.view(-1, input.size(-1)) if input.dim() > 2 else input
			_target = target.view(-1, 1)
			model_prob = self.weight.repeat(_target.size(0), 1)
			model_prob.scatter_(1, _target, self.conf)

			if isinstance(self.ignore_index, (list, tuple)):
				model_prob.masked_fill_(eq_indexes(_target, self.ignore_index), 0.0)
			elif self.ignore_index >= 0:
				model_prob.masked_fill_(_target == self.ignore_index, 0.0)

			rs = kl_div(_input, model_prob, reduction=self.reduction)

			return rs.view(input.size()) if self.reduction == 'none' and target.dim() > 1 else rs

class NLLLoss(NLLLossBase):

	def forward(self, input, target):

		rs = nll_loss(input.view(-1, input.size(-1)), target.view(-1), weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

		return rs.view(input.size()) if self.reduction == 'none' and target.dim() > 1 else rs

class RankingLoss(_Loss):

	# input: (batch size)
	# target: (batch size)
	def forward(self, input, target):

		loss = input * target
		if self.reduction == 'mean':
			loss = loss / loss.numel()

		return loss

class MultiLabelSmoothingLoss(_Loss):

	def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction='mean', forbidden_index=-1, fast_mode=use_fast_loss):

		super(MultiLabelSmoothingLoss, self).__init__()

		self.fast_mode, self.reduction = fast_mode, reduction

		fbil_common = set()
		if isinstance(ignore_index, (list, tuple)):
			tmp = []
			for _tmp in ignore_index:
				if (_tmp >= 0) and (_tmp not in tmp):
					tmp.append(_tmp)
					if _tmp not in fbil_common:
						fbil_common.add(_tmp)
			_nid = len(tmp)
			if _nid > 0:
				self.ignore_index = tuple(tmp) if _nid > 1 else tmp[0]
			else:
				self.ignore_index = ignore_index[0] if len(ignore_index) > 0 else -1
		else:
			self.ignore_index = ignore_index
			if (ignore_index >= 0) and (ignore_index not in fbil_common):
					fbil_common.add(ignore_index)

		if fast_mode:
			self.smoothing_value = label_smoothing / (nclass - 1)
			self.conf = 1.0 - label_smoothing - self.smoothing_value
		else:
			fbil = []
			for fbilu in forbidden_index:
				tmp = set()
				if isinstance(fbilu, (list, tuple)):
					for fi in fbilu:
						if (fi >= 0) and (fi not in tmp):
							tmp.add(fi)
				else:
					if fbilu is not None and fbilu >= 0:
						tmp.add(forbidden_index)
				tmp |= fbil_common
				fbil.append(tmp)

			_weight = []
			for fbilu in fbil:
				smoothing_value = label_smoothing / (nclass - 1 - len(fbilu))
				_tmp_w = torch.full((nclass,), smoothing_value)
				_tmp_w.index_fill_(0, torch.tensor(tuple(fbilu), dtype=torch.long, device=_tmp_w.device), 0.0)
				_weight.append(_tmp_w)
			self.register_buffer("weight", torch.stack(_weight, 0).unsqueeze(1))
			self.conf = 1.0 - label_smoothing

	def forward(self, input, target, lang_id=0):

		if self.fast_mode:

			return fast_label_smoothing_loss(input, target, self.ignore_index, self.conf, self.smoothing_value, self.reduction)
		else:
			_input = input.view(-1, input.size(-1)) if input.dim() > 2 else input
			_target = target.view(-1, 1)

			model_prob = self.weight[lang_id].repeat(_target.size(0), 1)
			model_prob.scatter_(1, _target, self.conf)

			if isinstance(self.ignore_index, (list, tuple)):
				model_prob.masked_fill_(eq_indexes(_target, self.ignore_index), 0.0)
			elif self.ignore_index >= 0:
				model_prob.masked_fill_(_target == self.ignore_index, 0.0)

			rs = kl_div(_input, model_prob, reduction=self.reduction)

			return rs.view(input.size()) if self.reduction == 'none' and target.dim() > 1 else rs

class ReducedLabelSmoothingLoss(LabelSmoothingLoss):

	def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction='mean', forbidden_index=-1, fast_mode=use_fast_loss, reduce_dim=None):

		super(ReducedLabelSmoothingLoss, self).__init__(nclass, label_smoothing=label_smoothing, ignore_index=ignore_index, reduction=reduction, forbidden_index=forbidden_index, fast_mode=fast_mode)

		self.reduce_dim = reduce_dim

	def forward(self, input, target):

		if self.reduce_dim is not None:
			input, target = clear_pad_mask([input, target], target.eq(0), [self.reduce_dim - 1, self.reduce_dim], mask_dim=self.reduce_dim, return_contiguous=True)[0]

		if self.fast_mode:

			return fast_label_smoothing_loss(input, target, self.ignore_index, self.conf, self.smoothing_value, self.reduction)
		else:
			_input = input.view(-1, input.size(-1)) if input.dim() > 2 else input
			_target = target.view(-1, 1)

			model_prob = self.weight.repeat(_target.size(0), 1)
			model_prob.scatter_(1, _target, self.conf)

			if isinstance(self.ignore_index, (list, tuple)):
				model_prob.masked_fill_(eq_indexes(_target, self.ignore_index), 0.0)
			elif self.ignore_index >= 0:
				model_prob.masked_fill_(_target == self.ignore_index, 0.0)

			rs = kl_div(_input, model_prob, reduction=self.reduction)

			return rs.view(input.size()) if self.reduction == 'none' and target.dim() > 1 else rs
