#encoding: utf-8

def set_is_decoding(m, mode):

	for _ in m.modules():
		if hasattr(_, "is_decoding"):
			if isinstance(_.is_decoding, bool):
				_.is_decoding = mode
			else:
				_.is_decoding(mode)

	return m

class model_decoding:

	def __init__(self, net, **kwargs):

		self.net = net

	def __enter__(self):

		return set_is_decoding(self.net, True)

	def __exit__(self, *inputs, **kwargs):

		set_is_decoding(self.net, False)
