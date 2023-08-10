#encoding: utf-8

from utils.data import inf_data_generator
from utils.random import multinomial

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

	def __init__(self, task_weight, task_weight_T, ntrain, train_taskl, nsample=None, **kwargs):

		self.generator = sample_iter(task_weight, task_weight_T, ntrain, train_taskl)
		self.nsample = nsample

	def generate(self, nsample=None):

		return [next(self.generator) for i in range(self.nsample if nsample is None else nsample)]

class balance_loader:

	def __init__(self, tls, sfunc=min):

		self.tls = tls
		self.imax = len(tls)
		self.imin = - (self.imax + 1)
		self.ndata = self.imax * sfunc(len(_) for _ in self.tls)
		self.dg = [inf_data_generator(_) for _ in self.tls]
		self.c = [0 for _ in range(self.imax)]

	def get_one(self):

		_im, _vm = 0, self.c[0]
		for _i, _v in enumerate(self.c):
			if _v < _vm:
				_im, _vm = _i, _v

		return _im, next(self.dg[_im])

	def __call__(self, ndata=None):

		for _ in range(self.ndata if ndata is None else ndata):
			yield self.get_one()

	def update(self, i, v=0):

		if (i < self.imax) and (i > self.imin) and (v > 0):
			_ = self.c[i] + v
			self.c = [0 if _i == i else (_v - _) for _i, _v in enumerate(self.c)]
