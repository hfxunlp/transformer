#encoding: utf-8

def parse_none(vin, value):

	return value if vin is None else vin

def parse_double_value_tuple(vin):

	if isinstance(vin, (list, tuple,)):
		return vin[0], vin[-1]
	else:
		return vin, vin
