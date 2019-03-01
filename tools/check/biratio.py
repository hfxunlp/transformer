#encoding: utf-8

import sys

def handle(srcfs, srcft):

	def ratio_bilingual(ls, lt):
		if ls > lt:
			return float(ls) / float(lt)
		else:
			return float(lt) / float(ls)

	bmaxratio = 0.0
	omaxratio = 0.0
	bmeanratio = 0.0
	omeanratio = 0.0
	ndata = 0
	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft:
		for sline, tline in zip(fs, ft):
			sline, tline = sline.strip(), tline.strip()
			if sline and tline:
				sline, tline = sline.decode("utf-8"), tline.decode("utf-8")
				bratio = ratio_bilingual(len(sline.split()), len(tline.split()))
				if bratio > bmaxratio:
					bmaxratio = bratio
				bmeanratio += bratio
				oratio = ratio_bilingual(len(sline.replace("@@ ", "").split()), len(tline.replace("@@ ", "").split()))
				if oratio > omaxratio:
					omaxratio = oratio
				omeanratio += oratio

				ndata += 1

	ndata = float(ndata)
	bmeanratio /= ndata
	omeanratio /= ndata

	print("Max/mean/adv bpe bilingual ratio is: %.3f / %.3f / %.3f\nMax/mean/adv original bilingual ratio is: %.3f / %.3f / %.3f" % (bmaxratio, bmeanratio, min(bmaxratio, bmeanratio * 2.5) + 0.001, omaxratio, omeanratio, min(omaxratio, omeanratio * 2.5) + 0.001))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
