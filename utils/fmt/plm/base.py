#encoding: utf-8

def fix_parameter_name(din):

	rs = {}
	for k, v in din.items():
		if k.endswith(".gamma"):
			k = k[:-6] + ".weight"
		elif k.endswith(".beta"):
			k = k[:-5] + ".bias"
		rs[k] = fix_parameter_name(v) if isinstance(v, dict) else v

	return rs
