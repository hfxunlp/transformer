#encoding: utf-8

from utils.random import multinomial
from utils.data import inf_data_generator

def T_normalize(wl, T):

	_t = 1.0 / T
	_tmp = [_wu ** _t for _wu in wl]
	_s = sum(_tmp)

	return [_tu / _s for _tu in _tmp]

def sample_iter(wl, T, ntrain, taskl):

	samples = {}
	for i, (nd, task,) in enumerate(zip(ntrain, taskl)):
		samples[i] = (task, inf_data_generator(str(i) for i in range(nd)),)
	pl = T_normalize(wl, T)
	while True:
		task, dg = samples[multinomial(pl, s=1.0)]
		yield next(dg), task

class data_sampler:

	def __init__(self, task_weight, task_weight_T, ntrain, train_taskl, nsample=None):

		self.generator = sample_iter(task_weight, task_weight_T, ntrain, train_taskl)
		self.nsample = nsample

	def generate(self, nsample=None):

		return [next(self.generator) for i in range(self.nsample if nsample is None else nsample)]
