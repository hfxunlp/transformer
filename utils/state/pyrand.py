#encoding: utf-8

from random import getstate, setstate

class PyRandomState:

	def state_dict(self):

		return {"random_state": getstate()}

	def load_state_dict(self, dictin):

		if "random_state" in dictin:
			setstate(dictin["random_state"])
