#encoding: utf-8

from math import log

def arcsigmoid(x):

	return -log((1.0/x)-1.0)
