#encoding: utf-8

from torch import nn
from modules.base import Linear
from modules.act import Custom_Act, LGLU, get_act, GELU
from modules.dropout import Dropout

from modules.TA import PositionwiseFF as PositionwiseFFBase

from cnfg.plm.bert.ihyp import *

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, use_glu=use_glu, **kwargs)

		if (use_glu is not None) and (_hsize % 2 == 1):
			_hsize += 1

		_ = [Linear(isize, _hsize)]
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else GELU(), Linear(_hsize, isize, bias=enable_bias)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
				_.append(LGLU())
			_.append(Linear(_hsize // 2, isize, bias=enable_bias))
		if dropout > 0.0:
			_.append(Dropout(dropout, inplace=True))
		self.net = nn.Sequential(*_)
