#encoding: utf-8

import torch

class THRandomState:

	def __init__(self, use_cuda=True, **kwargs):

		self.use_cuda = torch.cuda.is_available() and use_cuda

	def state_dict(self):

		rsd = {"random_state": torch.get_rng_state()}
		if self.use_cuda:
			rsd["cuda_random_state"] = torch.cuda.get_rng_state_all()

		return rsd

	def load_state_dict(self, dictin):

		if "random_state" in dictin:
			torch.set_rng_state(dictin["random_state"])
		if self.use_cuda and "cuda_random_state" in dictin:
			torch.cuda.set_rng_state_all(dictin["cuda_random_state"])
