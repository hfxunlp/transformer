#encoding: utf-8

class Holder:

	def __init__(self, **kwargs):

		self.nets = kwargs

	def state_dict(self, update=True, **kwargs):

		if len(kwargs) > 0:
			rsd = self.nets if update else self.nets.copy()
			rsd.update(kwargs)

		return {k: v.state_dict() if hasattr(v, "state_dict") else v for k, v in rsd.items()}

	def load_state_dict(self, dictin):

		left = {}
		for k, v in dictin.items():
			if k in self.nets:
				if hasattr(self.nets[k], "load_state_dict"):
					self.nets[k].load_state_dict(v)
				else:
					self.nets[k] = v
			else:
				left[k] = v

		return left
