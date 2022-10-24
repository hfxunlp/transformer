#encoding: utf-8

from json import loads, dumps

def dumpf(obj, fname):

	_ = dumps(obj)
	with open(fname, "wb") as f:
		f.write(_.encode("utf-8"))

def loadf(fname, print_func=print):

	with open(fname, "rb") as f:
		_ = f.read()
	try:
		return loads(_.decode("utf-8"))
	except Exception as e:
		if print_func is not None:
			print_func(e)
		return None
