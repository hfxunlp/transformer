#encoding: utf-8

import sys

# cratio: number of "@@" ended tokens / number of all tokens
# bratio: number of bpe tokens / number of tokens before bpe processing
# sratio: number of tokens seperated by bpe / number of tokens before bpe processing
# pratio: max(source length, target length) / min(source length, target length) length after bpe processing
# oratio: same as pratio but before bpe processing
# num_rules_drop: choose from [1, 6], fewer data will be droped with larger value, none data would be droped if it was set to 6

from utils.fmt.base import sys_open

def handle(srcfs, srcfm, srcft, tgtfs, tgtfm, tgtft, cratio=0.8, bratio=5.0, sratio=0.8, pratio=3.0, oratio=3.0, num_rules_drop=1):

	def legal_mono(strin, cratio, bratio, sratio):
		ntokens = nchars = nsp = nrule = 0
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
		ntokens = float(ntokens)
		lorigin = float(len(strin.replace("@@ ", "").split()))
		if float(nchars) / ntokens > cratio:
			nrule += 1
		if ntokens / lorigin > bratio:
			nrule += 1
		if float(nsp) / lorigin > sratio:
			nrule += 1
		return nrule, ntokens, lorigin

	def legal(strins, strint, cratio, bratio, sratio, pratio, oratio, num_rules_drop):

		def ratio_bilingual(ls, lt):
			return ls / lt if ls > lt else lt / ls

		ls, lens, lenso = legal_mono(strins, cratio, bratio, sratio)
		lt, lent, lento = legal_mono(strint, cratio, bratio, sratio)
		nrule = max(ls, lt)
		if ratio_bilingual(lens, lent) > pratio:
			nrule += 1
		if ratio_bilingual(lenso, lento) > oratio:
			nrule += 1
		return nrule < num_rules_drop

	ens = "\n".encode("utf-8")

	with sys_open(srcfs, "rb") as fs, sys_open(srcfm, "rb") as fm, sys_open(srcft, "rb") as ft, sys_open(tgtfs, "wb") as fsw, sys_open(tgtfm, "wb") as fmw, sys_open(tgtft, "wb") as ftw:
		total = keep = 0
		if num_rules_drop > 0:
			for ls, lm, lt in zip(fs, fm, ft):
				ls, lm, lt = ls.strip(), lm.strip(), lt.strip()
				if ls and lm and lt:
					ls, lm, lt = ls.decode("utf-8"), lm.decode("utf-8"), lt.decode("utf-8")
					if (num_rules_drop > 5) or legal(ls, lt, cratio, bratio, sratio, pratio, oratio, num_rules_drop):
						fsw.write(ls.encode("utf-8"))
						fsw.write(ens)
						fmw.write(lm.encode("utf-8"))
						fmw.write(ens)
						ftw.write(lt.encode("utf-8"))
						ftw.write(ens)
						keep += 1
					total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), float(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12]))
