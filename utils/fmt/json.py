#encoding: utf-8

from json import dumps, loads

from utils.fmt.base import sys_open

def dumpf(obj, fname):

	_ = dumps(obj)
	with sys_open(fname, "wb") as f:
		f.write(_.encode("utf-8"))

def loadf(fname, print_func=print):

	with sys_open(fname, "rb") as f:
		_ = f.read()
	try:
		return loads(_.decode("utf-8"))
	except Exception as e:
		if print_func is not None:
			print_func(e)
		return None
