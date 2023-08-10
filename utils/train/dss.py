#encoding: utf-8

from random import sample

def dynamic_sample(incd, dss_ws, dss_rm):

	rd = {}
	for k, v in incd.items():
		if v in rd:
			rd[v].append(k)
		else:
			rd[v] = [k]
	incs = list(rd.keys())
	incs.sort(reverse=True)
	_full_rl = []
	for v in incs:
		_full_rl.extend(rd[v])

	return _full_rl[:dss_ws] + sample(_full_rl[dss_ws:], dss_rm) if dss_rm > 0 else _full_rl[:dss_ws]
