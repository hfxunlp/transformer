#encoding: utf-8

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.modules.loss import NLLLoss as NLLLossBase

import torch.nn.functional as F

from utils.base import clear_pad_mask

"""	from: Rethinking the Inception Architecture for Computer Vision (https://arxiv.org/abs/1512.00567)
	With label smoothing, KL-divergence between q_{smoothed ground truth prob.}(w) and p_{prob. computed by model}(w) is minimized.
"""

class LabelSmoothingLoss(_Loss):

	def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction='mean', forbidden_index=-1):

		super(LabelSmoothingLoss, self).__init__()

		fbil = set()
		if isinstance(forbidden_index, (list, tuple)):
			for fi in forbidden_index:
				if (fi >= 0) and (fi not in fbil):
					fbil.add(fi)
		else:
			if forbidden_index is not None and forbidden_index >= 0:
				fbil.add(forbidden_index)

		if isinstance(ignore_index, (list, tuple)):
			tmp = []
			for _tmp in ignore_index:
				if (_tmp >= 0) and (_tmp not in tmp):
					tmp.append(_tmp)
					if _tmp not in fbil:
						fbil.add(_tmp)
			_nid = len(tmp)
			if _nid > 0:
				if _nid > 1:
					self.ignore_index = tuple(tmp)
				else:
					self.ignore_index = tmp[0]
			else:
				self.ignore_index = ignore_index[0] if len(ignore_index) > 0 else -1
		else:
			self.ignore_index = ignore_index
			if (ignore_index >= 0) and (ignore_index not in fbil):
				fbil.add(ignore_index)

		smoothing_value = label_smoothing / (nclass - 1 - len(fbil))
		weight = torch.full((nclass,), smoothing_value)
		weight.index_fill_(0, torch.tensor(tuple(fbil), dtype=torch.long, device=weight.device), 0.0)
		self.register_buffer("weight", weight.unsqueeze(0))

		self.reduction = reduction
		self.conf = 1.0 - label_smoothing

	# output: (batch size, num_classes)
	# target: (batch size)
	# they will be flattened automatically if the dimension of output is larger than 2.

	def forward(self, output, target):

		_output = output.view(-1, output.size(-1)) if output.dim() > 2 else output

		_target = target.view(-1, 1)

		model_prob = self.weight.repeat(_target.size(0), 1)
		model_prob.scatter_(1, _target, self.conf)

		if isinstance(self.ignore_index, (list, tuple)):
			model_prob.masked_fill_(torch.stack([_target == _tmp for _tmp in self.ignore_index]).int().sum(0).gt(0), 0.0)
		elif self.ignore_index >= 0:
			model_prob.masked_fill_(_target == self.ignore_index, 0.0)

		return F.kl_div(_output, model_prob, reduction=self.reduction)

class NLLLoss(NLLLossBase):

	def forward(self, input, target):

		isize = input.size()

		return F.nll_loss(input.view(-1, isize[-1]), target.view(-1), weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction).view(isize[:-1])

class RankingLoss(_Loss):

	# output: (batch size)
	# target: (batch size)
	def forward(self, output, target):

		loss = output * target
		if self.reduction == 'mean':
			loss = loss / loss.numel()

		return loss

class MultiLabelSmoothingLoss(_Loss):

	def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction='mean', forbidden_index=-1):

		super(MultiLabelSmoothingLoss, self).__init__()

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
			fbil.append(tmp)

		if isinstance(ignore_index, (list, tuple)):
			tmp = []
			for _tmp in ignore_index:
				if (_tmp >= 0) and (_tmp not in tmp):
					tmp.append(_tmp)
					for fbilu in fbil:
						if _tmp not in fbilu:
							fbilu.add(_tmp)
			_nid = len(tmp)
			if _nid > 0:
				if _nid > 1:
					self.ignore_index = tuple(tmp)
				else:
					self.ignore_index = tmp[0]
			else:
				self.ignore_index = ignore_index[0] if len(ignore_index) > 0 else -1
		else:
			self.ignore_index = ignore_index
			if (ignore_index >= 0):
				for fbilu in fbil:
					if ignore_index not in fbilu:
						fbilu.add(ignore_index)

		_weight = []
		for fbilu in fbil:
			smoothing_value = label_smoothing / (nclass - 1 - len(fbilu))
			_tmp_w = torch.full((nclass,), smoothing_value)
			_tmp_w.index_fill_(0, torch.tensor(tuple(fbilu), dtype=torch.long, device=_tmp_w.device), 0.0)
			_weight.append(_tmp_w)
		self.register_buffer("weight", torch.stack(_weight, 0).unsqueeze(1))

		self.reduction = reduction

		self.conf = 1.0 - label_smoothing

	def forward(self, output, target, lang_id=0):

		_output = output.view(-1, output.size(-1)) if output.dim() > 2 else output

		_target = target.view(-1, 1)

		model_prob = self.weight[lang_id].repeat(_target.size(0), 1)
		model_prob.scatter_(1, _target, self.conf)

		if isinstance(self.ignore_index, (list, tuple)):
			model_prob.masked_fill_(torch.stack([_target == _tmp for _tmp in self.ignore_index]).int().sum(0).gt(0), 0.0)
		elif self.ignore_index >= 0:
			model_prob.masked_fill_(_target == self.ignore_index, 0.0)

		return F.kl_div(_output, model_prob, reduction=self.reduction)

class ReducedLabelSmoothingLoss(LabelSmoothingLoss):

	def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction='mean', forbidden_index=-1, reduce_dim=None):

		super(ReducedLabelSmoothingLoss, self).__init__(nclass, label_smoothing, ignore_index, reduction, forbidden_index)

		self.reduce_dim = reduce_dim

	def forward(self, output, target):

		if self.reduce_dim is not None:
			output, target = clear_pad_mask([output, target], target.eq(0), [self.reduce_dim - 1, self.reduce_dim], mask_dim=self.reduce_dim, return_contiguous=True)[0]

		_output = output.view(-1, output.size(-1)) if output.dim() > 2 else output

		_target = target.view(-1, 1)

		model_prob = self.weight.repeat(_target.size(0), 1)
		model_prob.scatter_(1, _target, self.conf)

		if isinstance(self.ignore_index, (list, tuple)):
			model_prob.masked_fill_(torch.stack([_target == _tmp for _tmp in self.ignore_index]).int().sum(0).gt(0), 0.0)
		elif self.ignore_index >= 0:
			model_prob.masked_fill_(_target == self.ignore_index, 0.0)

		return F.kl_div(_output, model_prob, reduction=self.reduction)
