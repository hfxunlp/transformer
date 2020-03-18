#encoding: utf-8

import os
from os.path import sep
from subprocess import PIPE, Popen

perl_exec = "perl"
moses_scripts = os.environ.get('moses_scripts')

if not moses_scripts.endswith(sep):
	moses_scripts += sep

class ProcessWrapper:

	def __init__(self, cmd=[]):

		self.process = None
		self.cmd = cmd

	def start(self, stdin=PIPE, stdout=PIPE):

		if self.process:
			raise Exception("Process is already running")
		self.process = Popen(self.cmd, stdin=stdin, stdout=stdout)

	def __del__(self):

		if self.process:
			self.process.terminate()

class LineProcessor(ProcessWrapper):

	def __call__(self, input):

		self.process.stdin.write(("%s\n" % input.strip()).encode("utf-8", "ignore"))
		self.process.stdin.flush()

		return self.process.stdout.readline().strip().decode("utf-8", "ignore")

class BatchProcessor(ProcessWrapper):

	def __call__(self, input):

		if isinstance(input, (list, tuple)):
			rs = []

			for inputu in input:
				self.process.stdin.write(("%s\n" % inputu.strip()).encode("utf-8", "ignore"))
				self.process.stdin.flush()
				rs.append(self.process.stdout.readline().strip().decode("utf-8", "ignore"))
		else:
			self.process.stdin.write(("%s\n" % input.strip()).encode("utf-8", "ignore"))
			self.process.stdin.flush()
			rs = self.process.stdout.readline().strip().decode("utf-8", "ignore")

		return rs

class SentenceSplitter(ProcessWrapper):
	"""Wrapper for standard Moses sentence splitter."""

	def __init__(self, lang):

		ssplit_cmd = moses_scripts + sep.join(("ems", "support", "split-sentences.perl"))
		self.cmd = [perl_exec, ssplit_cmd, "-b", "-q", "-l", lang]
		self.process = None
		self.start()

	def __call__(self, input):

		self.process.stdin.write((input.strip() + "\n<P>\n").encode("utf-8", "ignore"))
		self.process.stdin.flush()
		x = self.process.stdout.readline().strip().decode("utf-8", "ignore")
		ret = []
		while x != '<P>' and x != '':
			ret.append(x)
			x = self.process.stdout.readline().strip().decode("utf-8", "ignore")

		return ret

class Pretokenizer(BatchProcessor):
	"""Pretokenizer wrapper.
	The pretokenizer fixes known issues with the input.
	"""
	def __init__(self, lang):

		pretok_cmd = moses_scripts + sep.join(("tokenizer", "pre-tokenizer.perl"))
		self.cmd = [perl_exec, pretok_cmd, "-b", "-q", "-l", lang]
		self.process = None
		self.start()

class Tokenizer(BatchProcessor):
	"""Tokenizer wrapper.
	The pretokenizer fixes known issues with the input.
	"""
	# default args: ["-a", "-no-escape"]
	def __init__(self, lang, args=["-a"]):

		tok_cmd = moses_scripts + sep.join(("tokenizer", "tokenizer.perl"))
		self.cmd = [perl_exec, tok_cmd, "-b", "-q", "-l", lang] + args
		self.process = None
		self.start()

class Normalizepunctuation(BatchProcessor):

	def __init__(self, lang):

		tok_cmd = moses_scripts + sep.join(("tokenizer", "normalize-punctuation.perl"))
		self.cmd = [perl_exec, tok_cmd, "-b", "-q", "-l", lang]
		self.process = None
		self.start()

class Truecaser(BatchProcessor):
	"""Truecaser wrapper."""
	def __init__(self, model):

		truecase_cmd = moses_scripts + sep.join(("recaser", "truecase.perl"))
		self.cmd = [perl_exec, truecase_cmd, "-b", "--model", model]
		self.process = None
		self.start()

class Detruecaser(BatchProcessor):

	def __init__(self):

		truecase_cmd = moses_scripts + sep.join(("recaser", "detruecase.perl"))
		self.cmd = [perl_exec, truecase_cmd, "-b"]
		self.process = None
		self.start()

class Detokenizer(BatchProcessor):

	# default args: ["-a", "-no-escape"]
	def __init__(self, lang):

		tok_cmd = moses_scripts + sep.join(("tokenizer", "detokenizer.perl"))
		self.cmd = [perl_exec, tok_cmd, "-q", "-b", "-l", lang]
		self.process = None
		self.start()
