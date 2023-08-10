#encoding: utf-8

import codecs
import re
from random import random

class BPE(object):

	def __init__(self, codes, merges=-1, separator="@@", vocab=None, glossaries=None, **kwargs):

		codes.seek(0)
		offset=1

		# check version information
		firstline = codes.readline()
		if firstline.startswith("#version:"):
			self.version = tuple([int(x) for x in re.sub(r"(\.0+)*$","", firstline.split()[-1]).split(".")])
			offset += 1
		else:
			self.version = (0, 1)
			codes.seek(0)

		self.bpe_codes = [tuple(item.strip("\r\n ").split(" ")) for (n, item) in enumerate(codes) if (n < merges or merges == -1)]

		# some hacking to deal with duplicates (only consider first instance)
		self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

		self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

		self.separator = separator

		self.vocab = vocab

		self.glossaries = glossaries if glossaries else []

		self.glossaries_regex = re.compile("^({})$".format("|".join(glossaries))) if glossaries else None

		self.cache = {}

	def process_line(self, line, dropout=0):
		"""segment line, dealing with leading and trailing whitespace"""

		out = ""

		leading_whitespace = len(line)-len(line.lstrip("\r\n "))
		if leading_whitespace:
			out += line[:leading_whitespace]

		out += self.segment(line, dropout)

		trailing_whitespace = len(line)-len(line.rstrip("\r\n "))
		if trailing_whitespace and trailing_whitespace != len(line):
			out += line[-trailing_whitespace:]

		return out

	def segment(self, sentence, dropout=0):
		"""segment single sentence (whitespace-tokenized string) with BPE encoding"""
		segments = self.segment_tokens(sentence.strip("\r\n ").split(" "), dropout)
		return " ".join(segments)

	def segment_tokens(self, tokens, dropout=0):
		"""segment a sequence of tokens with BPE encoding"""
		output = []
		for word in tokens:
			# eliminate double spaces
			if not word:
				continue
			new_word = [out for segment in self._isolate_glossaries(word) for out in encode(segment, self.bpe_codes, self.bpe_codes_reverse, self.vocab, self.separator, self.version, self.cache, self.glossaries_regex, dropout)]

			for item in new_word[:-1]:
				output.append(item + self.separator)
			output.append(new_word[-1])

		return output

	def _isolate_glossaries(self, word):
		word_segments = [word]
		for gloss in self.glossaries:
			word_segments = [out_segments for segment in word_segments for out_segments in isolate_glossary(segment, gloss)]
		return word_segments

def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries_regex=None, dropout=0):
	"""Encode word based on list of BPE merge operations, which are applied consecutively
	"""

	if not dropout and orig in cache:
		return cache[orig]

	if glossaries_regex and glossaries_regex.match(orig):
		cache[orig] = (orig,)
		return (orig,)

	if len(orig) == 1:
		return orig

	if version == (0, 1):
		word = list(orig) + ["</w>"]
	elif version == (0, 2): # more consistent handling of word-final segments
		word = list(orig[:-1]) + [orig[-1] + "</w>"]
	else:
		raise NotImplementedError

	while len(word) > 1:

		# get list of symbol pairs; optionally apply dropout
		pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not dropout or random() > dropout) and pair in bpe_codes]

		if not pairs:
			break

		#get first merge operation in list of BPE codes
		bigram = min(pairs)[2]

		# find start position of all pairs that we want to merge
		positions = [i for (rank,i,pair) in pairs if pair == bigram]

		i = 0
		new_word = []
		bigram = "".join(bigram)
		for j in positions:
			# merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
			if j < i:
				continue
			new_word.extend(word[i:j]) # all symbols before merged pair
			new_word.append(bigram) # merged pair
			i = j+2 # continue after merged pair
		new_word.extend(word[i:]) # add all symbols until end of word
		word = new_word

	# don"t print end-of-word symbols
	if word[-1] == "</w>":
		word = word[:-1]
	elif word[-1].endswith("</w>"):
		word[-1] = word[-1][:-4]

	word = tuple(word)
	if vocab:
		word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

	cache[orig] = word
	return word

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
	"""Recursively split segment into smaller units (by reversing BPE merges)
	until all units are either in-vocabulary, or cannot be split futher."""

	try:
		if final:
			left, right = bpe_codes[segment + "</w>"]
			right = right[:-4]
		else:
			left, right = bpe_codes[segment]
	except:
		#sys.stderr.write("cannot split {0} further.\n".format(segment))
		yield segment
		return

	if left + separator in vocab:
		yield left
	else:
		for item in recursive_split(left, bpe_codes, vocab, separator, False):
			yield item

	if (final and right in vocab) or (not final and right + separator in vocab):
		yield right
	else:
		for item in recursive_split(right, bpe_codes, vocab, separator, final):
			yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
	"""Check for each segment in word if it is in-vocabulary,
	and segment OOV segments into smaller units by reversing the BPE merge operations"""

	out = []

	for segment in orig[:-1]:
		if segment + separator in vocab:
			out.append(segment)
		else:
			for item in recursive_split(segment, bpe_codes, vocab, separator, False):
				out.append(item)

	segment = orig[-1]
	if segment in vocab:
		out.append(segment)
	else:
		for item in recursive_split(segment, bpe_codes, vocab, separator, True):
			out.append(item)

	return out

def read_vocabulary(vocab_file, threshold):
	"""read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
	"""

	vocabulary = set()

	for line in vocab_file:
		word, freq = line.strip("\r\n ").split(" ")
		freq = int(freq)
		if threshold == None or freq >= threshold:
			vocabulary.add(word)

	return vocabulary

def isolate_glossary(word, glossary):
	"""
	Isolate a glossary present inside a word.
	Returns a list of subwords. In which all "glossary" glossaries are isolated
	For example, if "USA" is the glossary and "1934USABUSA" the word, the return value is:
		["1934", "USA", "B", "USA"]
	"""
	# regex equivalent of (if word == glossary or glossary not in word)
	if re.match("^"+glossary+"$", word) or not re.search(glossary, word):
		return [word]
	else:
		segments = re.split(r"({})".format(glossary), word)
		segments, ending = segments[:-1], segments[-1]
		segments = list(filter(None, segments)) # Remove empty strings in regex group.
		return segments + [ending.strip("\r\n ")] if ending != "" else segments

class BPERemover:

	def __call__(self, input, **kwargs):

		if isinstance(input, (list, tuple,)):
			rs = []
			for inputu in input:
				rs.append(inputu.replace("@@ ", ""))
			return rs
		else:
			return input.replace("@@ ", "")

class BPEApplier:

	def __init__(self, codesf, bpe_vcb=None, vocabulary_threshold=None, separator="@@", merges=-1, glossaries=None, **kwargs):

		if bpe_vcb is not None:
			vocabulary = read_vocabulary(codecs.open(bpe_vcb, encoding="utf-8"), vocabulary_threshold)
		else:
			vocabulary = None
		if glossaries is not None:
			glossaries = [g.decode("utf-8") for g in glossaries]
		self.bpe = BPE(codecs.open(codesf, encoding="utf-8"), merges, separator, vocabulary, glossaries)

	def __call__(self, input, **kwargs):

		if isinstance(input, (list, tuple,)):
			rs = []
			for inputu in input:
				rs.append(self.bpe.process_line(inputu))
			return rs
		else:
			return self.bpe.process_line(input)
