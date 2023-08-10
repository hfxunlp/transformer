#encoding: utf-8

try:
	from opencc import OpenCC
except Exception as e:
	print(e)
	OpenCC = None
	from utils.func import identity_func

build_func = identity_func if OpenCC is None else (lambda task: OpenCC("%s.json" % task).convert)

t2s_func = build_func("t2s")

def vcb_filter_func_opencc(din, func=t2s_func):

	rsd = {}
	for k, v in din.items():
		_ = func(k)
		_k = k if (_ == k) or (_ not in din) else _
		rsd[_k] = rsd.get(_k, 0) + v

	return rsd

vcb_filter_func = identity_func if OpenCC is None else vcb_filter_func_opencc
