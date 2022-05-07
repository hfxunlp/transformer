#encoding: utf-8

# portal from fairseq: https://github.com/pytorch/fairseq/blob/master/scripts/spm_encode.py

import sys
from argparse import ArgumentParser
from sentencepiece import SentencePieceProcessor

def main():
	parser = ArgumentParser()
	parser.add_argument("--model", required=True, help="sentencepiece model to use for decoding")
	parser.add_argument("--input", default="-", help="input file to decode")
	parser.add_argument("--input_format", choices=["piece", "id"], default="piece")
	args = parser.parse_args()

	sp = SentencePieceProcessor()
	sp.Load(args.model)

	if args.input_format == "piece":

		def decode(l):
			return "".join(sp.DecodePieces(l))

	elif args.input_format == "id":

		def decode(l):
			return "".join(sp.DecodeIds(l))

	def tok2int(tok):
		# remap reference-side <unk> to 0
		return int(tok) if tok != "<unk>" else 0

	with sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8") as h:
		if args.input_format == "id":
			for line in h:
				print(decode(list(map(tok2int, line.rstrip().split()))))
		elif args.input_format == "piece":
			for line in h:
				print(decode(line.rstrip().split()))

if __name__ == "__main__":
	main()
