#encoding: utf-8

import torch
from math import sqrt

from utils.torch.comp import torch_no_grad

from cnfg.ihyp import ieps_default

def prep_cos(og, ng):

	return (og * ng).sum(), og.pow(2).sum(), ng.pow(2).sum()

def cos_acc_pg(old_pg, new_pg, ieps=ieps_default):

	with torch_no_grad():
		on, o, n = zip(*[prep_cos(ou, nu) for ou, nu in zip(old_pg, new_pg)])
		sim = (torch.stack(on, 0).sum() / (torch.stack(o, 0).sum() * torch.stack(n, 0).sum()).sqrt().add(ieps)).item()

	return min(max(-1.0, sim), 1.0)
