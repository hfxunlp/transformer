#encoding: utf-8

import sys

def handle(srcfs, srcft, srcfg, tgtfs, tgtft, tgtfg):

	def clean(strin):
		rs = []
		for tmpu in strin.split():
			if tmpu:
				rs.append(tmpu)
		return " ".join(rs)

	ens = "\n".encode("utf-8")

	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft, open(srcfg, "rb") as fg, open(tgtfs, "wb") as fsw, open(tgtft, "wb") as ftw, open(tgtfg, "wb") as fgw:
		total = 0
		keep = 0
		for ls, lt, lg in zip(fs, ft, fg):
			ls, lt, lg = ls.strip(), lt.strip(), lg.strip()
			if ls and lt and lg:
				ls, lt, lg = clean(ls.decode("utf-8")), clean(lt.decode("utf-8")), clean(lg.decode("utf-8"))
				if lt != lg:
					fsw.write(ls.encode("utf-8"))
					fsw.write(ens)
					ftw.write(lt.encode("utf-8"))
					ftw.write(ens)
					fgw.write(lg.encode("utf-8"))
					fgw.write(ens)
					keep += 1
				total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
