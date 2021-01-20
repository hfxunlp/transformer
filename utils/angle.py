#encoding: utf-8

import torch
from math import sqrt

def prep_cos(og, ng):

	return (og * ng).sum(), og.pow(2).sum(), ng.pow(2).sum()

def cos_acc_pg(old_pg, new_pg):

	with torch.no_grad():
		on, o, n = zip(*[prep_cos(ou, nu) for ou, nu in zip(old_pg, new_pg)])
		sim = (torch.stack(on, 0).sum() / (torch.stack(o, 0).sum() * torch.stack(n, 0).sum()).sqrt()).item()

	return sim
