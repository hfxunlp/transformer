#encoding: utf-8

from random import shuffle

def inf_data_generator(dlin, shuf=True):

	tmp = list(dlin)
	while True:
		if shuf:
			shuffle(tmp)
		yield from tmp
