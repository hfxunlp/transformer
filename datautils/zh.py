#encoding: utf-8

from pynlpir import nlpir

splitcode=set([u"。", u"？", u"！", u"；", u"\n"])

class SentenceSplitter:
	"""Wrapper for standard Moses sentence splitter."""

	def __init__(self, splc=splitcode, **kwargs):

		self.splc = splc

	def __call__(self, input, **kwargs):

		rs = []
		ind = lind = 0
		for stru in input:
			if stru in self.splc:
				rs.append(input[lind:ind+1])
				ind += 1
				lind = ind
			else:
				ind += 1
		if lind < ind:
			rs.append(input[lind:])

		return rs

class Tokenizer:

	def __init__(self):

		self.start()

	def start(self):

		nlpir.Init(nlpir.PACKAGE_DIR.encode("utf-8"), nlpir.UTF8_CODE, "".encode("utf-8"))

	def __del__(self):

		nlpir.Exit()

	def __call__(self, input, **kwargs):

		def clear_tag(strin):

			tmp = strin.split()
			rs = []
			for tmpu in tmp:
				if tmpu:
					ind = tmpu.rfind("/")
					if ind > 0:
						rs.append(tmpu[:ind])
					else:
						rs.append(tmpu)

			return " ".join(rs)

		if not isinstance(input, (list, tuple,)):
			input = [input]

		rs = []
		for inputu in input:
			try:
				_tmp = nlpir.ParagraphProcess(inputu.encode("utf-8", "ignore"), 1)
			except:
				_tmp = ""
			rs.append(clear_tag(_tmp.decode("utf-8", "ignore")))

		return rs

class Detokenizer:

	def __call__(self, input, **kwargs):

		if not isinstance(input, (list, tuple,)):
			input = [input]
		rs = []
		for inputu in input:
			_tmp = []
			for tmpu in inputu.strip().split():
				if tmpu:
					_tmp.append(tmpu)
			rs.append("".join(_tmp))

		return rs
