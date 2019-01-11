#encoding: utf-8

import sys

def handle(srcfs, srcft):

	def getfratio(fname):

		def get_ratio(strin):
			ntokens = 0
			nchars = 0
			nsp = 0
			pbpe = False
			for tmpu in strin.split():
				if tmpu:
					if tmpu.endswith("@@"):
						nchars += 1
						if not pbpe:
							pbpe = True
							nsp += 1
					elif pbpe:
						pbpe = False
					ntokens += 1
			lorigin = float(len(strin.replace("@@ ", "").split()))
			ntokens = float(ntokens)
			return float(nchars) / ntokens, ntokens / lorigin, float(nsp) / lorigin

		with open(fname, "rb") as fs:

			maxratioc = 0.0
			maxratiob = 0.0
			maxratios = 0.0

			for line in fs:
				line = line.strip()
				if line:
					ratioc, ratiob, ratios = get_ratio(line.decode("utf-8"))
					if ratioc > maxratioc:
						maxratioc = ratioc
					if ratiob > maxratiob:
						maxratiob = ratiob
					if ratios > maxratios:
						maxratios = ratios

		return maxratioc, maxratiob, maxratios

	mrsc, mrsb, mrss = getfratio(srcfs)
	mrtc, mrtb, mrts = getfratio(srcft)

	print("Maximum char ratio of source data: %f\nMaximum char ratio of target data: %f\nMaximum expand ratio of source data: %f\nMaximum expand ratio of target data: %f\nMaximum seperated ratio of source data: %f\nMaximum seperated ratio of target data: %f" % (mrsc, mrtc, mrsb, mrtb, mrss, mrts))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
