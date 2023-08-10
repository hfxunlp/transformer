#encoding: utf-8

from sacremoses import MosesDetokenizer, MosesDetruecaser, MosesPunctNormalizer, MosesTokenizer, MosesTruecaser

from utils.fmt.base import clean_list

class BatchProcessor():

	def __call__(self, input, **kwargs):

		return [self.process(inputu) for inputu in input] if isinstance(input, (list, tuple,)) else self.process(input)

	def process(self, input):

		return input

class Tokenizer(BatchProcessor):

	# default args: ["-a", "-no-escape"]
	def __init__(self, lang, args=["-a"], **kwargs):

		self.handler = MosesTokenizer(lang=lang)
		self.escape = not ("-no-escape" in args or "--no-escape" in args)
		self.aggressive = "-a" in args

	def process(self, input):

		return self.handler.tokenize(input, aggressive_dash_splits=self.aggressive, return_str=True, escape=self.escape)

class Normalizepunctuation(BatchProcessor):

	def __init__(self, lang, **kwargs):

		self.handler = MosesPunctNormalizer(lang=lang)

	def process(self, input):

		return self.handler.normalize(input)

class Truecaser(BatchProcessor):

	def __init__(self, model, **kwargs):

		self.handler = MosesTruecaser(load_from=model)

	def process(self, input):

		return self.handler.truecase(input.encode("utf-8", "ignore"), return_str=True).decode("utf-8", "ignore")

class Detruecaser(BatchProcessor):

	def __init__(self):

		self.handler = MosesDetruecaser()

	def process(self, input, is_headline=False):

		return self.handler.detruecase(input, is_headline=False, return_str=True)

class Detokenizer(BatchProcessor):

	def __init__(self, lang, **kwargs):

		self.handler = MosesDetokenizer(lang=lang)

	def process(self, input):

		return self.handler.detokenize(clean_list(input.split()))
