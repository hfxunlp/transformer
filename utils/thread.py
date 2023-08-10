#encoding: utf-8

from threading import Lock, Thread
from time import sleep

class LockHolder:

	def __init__(self, value=None):

		self.value = value
		self.lck = Lock()

	def __call__(self, *args):

		if args:
			with self.lck:
				self.value = args[0]
		else:
			with self.lck:
				return self.value

def start_thread(*args, **kwargs):

	_ = Thread(*args, **kwargs)
	_.start()

	return _

def thread_keeper_core(t, sleep_secs, *args, **kwargs):

	if t.is_alive():
		sleep(sleep_secs)
	else:
		t.join()
		t = start_thread(*args, **kwargs)

	return t

def thread_keeper(conditions, func, sleep_secs, *args, **kwargs):

	_conditions = tuple(conditions)
	_t = start_thread(*args, **kwargs)
	if len(_conditions) > 1:
		while func(_() for _ in _conditions):
			_t = thread_keeper_core(_t, sleep_secs, *args, **kwargs)
	else:
		_condition = _conditions[0]
		while _condition():
			_t = thread_keeper_core(_t, sleep_secs, *args, **kwargs)

def start_thread_with_keeper(conditions, func, sleep_secs, *args, **kwargs):

	return start_thread(target=thread_keeper, args=[conditions, func, sleep_secs, *args], kwargs=kwargs)
