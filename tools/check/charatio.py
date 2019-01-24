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

			meanratioc = 0.0
			meanratiob = 0.0
			meanratios = 0.0

			ndata = 0
			for line in fs:
				line = line.strip()
				if line:
					ratioc, ratiob, ratios = get_ratio(line.decode("utf-8"))
					if ratioc > maxratioc:
						maxratioc = ratioc
					meanratioc += ratioc
					if ratiob > maxratiob:
						maxratiob = ratiob
					meanratiob += ratiob
					if ratios > maxratios:
						maxratios = ratios
					meanratios += ratios
					ndata += 1

		ndata = float(ndata)
		meanratioc /= ndata
		meanratiob /= ndata
		meanratios /= ndata

		return maxratioc, meanratioc, maxratiob, meanratiob, maxratios, meanratios

	mrsc, _rsc, mrsb, _rsb, mrss, _rss = getfratio(srcfs)
	mrtc, _rtc, mrtb, _rtb, mrts, _rts = getfratio(srcft)

	print("Max/mean char ratio of source data: %.3f / %.3f\nMax/mean char ratio of target data: %.3f / %.3f\nMax/mean bpe ratio of source data: %.3f / %.3f\nMax/mean bpe ratio of target data: %.3f / %.3f\nMax/mean seperated ratio of source data: %.3f / %.3f\nMax/mean seperated ratio of target data: %.3f / %.3f" % (mrsc, _rsc, mrtc, _rtc, mrsb, _rsb, mrtb, _rtb, mrss, _rss, mrts, _rts))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
