#encoding: utf-8

from multiprocessing import Process
from time import sleep

def start_process(*args, **kwargs):

	_ = Process(*args, **kwargs)
	_.start()

	return _

def process_keeper_core(t, sleep_secs, *args, **kwargs):

	if t.is_alive():
		sleep(sleep_secs)
	else:
		t.join()
		t.close()
		t = start_process(*args, **kwargs)

	return t

def process_keeper(condition, sleep_secs, *args, **kwargs):

	_t = start_process(*args, **kwargs)
	while condition.value:
		_t = process_keeper_core(_t, sleep_secs, *args, **kwargs)
