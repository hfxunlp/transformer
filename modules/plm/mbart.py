#encoding: utf-8

import torch

from modules.base import CrossAttn as CrossAttnBase, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase

from cnfg.plm.mbart.ihyp import *

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, k_rel_pos=use_k_relative_position, uni_direction_reduction=False, is_left_to_right_reduction=True, zero_reduction=relpos_reduction_with_zeros, max_bucket_distance=0, sparsenorm=False, xseql=cache_len_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, k_rel_pos=k_rel_pos, uni_direction_reduction=uni_direction_reduction, is_left_to_right_reduction=is_left_to_right_reduction, zero_reduction=zero_reduction, max_bucket_distance=max_bucket_distance, sparsenorm=sparsenorm, xseql=xseql, **kwargs)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, sparsenorm=False, is_decoding=False, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, k_isize=k_isize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, sparsenorm=sparsenorm, **kwargs)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, **kwargs):

		super(PositionwiseFF, self).__init__(isize, hsize=hsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, use_glu=use_glu, **kwargs)
