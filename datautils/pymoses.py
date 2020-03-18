#encoding: utf-8

from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer, MosesTruecaser, MosesDetruecaser

from utils.fmt.base import clean_list

class BatchProcessor():

	def __call__(self, input):

		return [self.process(inputu) for inputu in input] if isinstance(input, (list, tuple)) else self.process(input)

	def process(self, input):

		return input

class Tokenizer(BatchProcessor):

	# default args: ["-a", "-no-escape"]
	def __init__(self, lang, args=["-a"]):

		self.handler = MosesTokenizer(lang=lang)
		self.escape = not ("-no-escape" in args or "--no-escape" in args)
		self.aggressive = "-a" in args

	def process(self, input):

		return self.handler.tokenize(input, aggressive_dash_splits=self.aggressive, return_str=True, escape=self.escape)

class Normalizepunctuation(BatchProcessor):

	def __init__(self, lang):

		self.handler = MosesPunctNormalizer(lang=lang)

	def process(self, input):

		return self.handler.normalize(input)

class Truecaser(BatchProcessor):

	def __init__(self, model):

		self.handler = MosesTruecaser(load_from=model)

	def process(self, input):

		return self.handler.truecase(input.encode("utf-8", "ignore")), return_str=True).decode("utf-8", "ignore")

class Detruecaser(BatchProcessor):

	def __init__(self):

		self.handler = MosesTruecaser()

	def process(self, input, is_headline=False):

		return self.handler.detruecase(input, is_headline=False, return_str=True)

class Detokenizer(BatchProcessor):

	def __init__(self, lang):

		self.handler = MosesDetokenizer(lang=lang)

	def process(self, input):

		return self.handler.detokenize(clean_list(input.split()))
