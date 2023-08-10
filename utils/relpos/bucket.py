#encoding: utf-8

import torch
from math import log

"""
relative postional encoding of T5, implementation of the transformers library for reference: https://github.com/huggingface/transformers/blob/v4.21.2/src/transformers/models/t5/modeling_t5.py#L374-L419

def _relative_position_bucket(length, num_buckets=32, max_distance=128, bidirectional=True):

	relative_position = torch.arange(length, dtype=torch.long, device=None)[None, :] - torch.arange(length, dtype=torch.long, device=None)[:, None]
	relative_buckets = 0
	if bidirectional:
		num_buckets //= 2
		relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
		relative_position = torch.abs(relative_position)
	else:
		relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
	max_exact = num_buckets // 2
	is_small = relative_position < max_exact

	relative_position_if_large = max_exact + (torch.log(relative_position.float() / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.long)
	relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))

	relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)

	return relative_buckets
"""

def build_rel_pos_bucket_map(k_rel_pos=32, max_len=128, uni_direction=False, device=None):

	_ = k_rel_pos + 1
	_thres = _ // 2
	_f_thres = float(_thres)
	_m = torch.arange(_thres, max_len, device=device).div(_f_thres).log().div(log(float(max_len) / _f_thres)).mul(float(_) - _f_thres).long().add_(_thres)

	return torch.cat((torch.arange(0, _thres, dtype=torch.long, device=device), _m,), dim=-1) if uni_direction else torch.cat((-_m.flip(-1), torch.arange(-_thres + 1, _thres, dtype=torch.long, device=device), _m,), dim=-1).add_(k_rel_pos)

def build_rel_pos_bucket_distance(length, max_len=128, uni_direction=False, device=None):

	_ = torch.arange(0, length, dtype=torch.long, device=device)
	_ = (_.unsqueeze(0) - _.unsqueeze(1))
	_max_side = max_len - 1

	return -(_.clamp(min=-_max_side, max=0)) if uni_direction else _.clamp(min=-_max_side, max=_max_side).add_(_max_side)

def map_rel_pos_bucket_distance(dis_map, distance):

	return dis_map.index_select(0, distance.view(-1)).view_as(distance)

def build_rel_pos_bucket(length, k_rel_pos=32, max_len=128, uni_direction=False, device=None, dis_map=None):

	return map_rel_pos_bucket_distance(build_rel_pos_bucket_map(k_rel_pos=k_rel_pos, max_len=max_len, uni_direction=uni_direction, device=device) if dis_map is None else dis_map, build_rel_pos_bucket_distance(length, max_len=max_len, uni_direction=uni_direction, device=dis_map.device if (device is None) and (dis_map is not None) else device))
