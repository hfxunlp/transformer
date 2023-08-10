#encoding: utf-8

import sys

from utils.fmt.base import get_bi_ratio, sys_open

def handle(srcfs, srcft):

	bmaxratio = 0.0
	omaxratio = 0.0
	bmeanratio = 0.0
	omeanratio = 0.0
	ndata = 0
	with sys_open(srcfs, "rb") as fs, sys_open(srcft, "rb") as ft:
		for sline, tline in zip(fs, ft):
			sline, tline = sline.strip(), tline.strip()
			if sline and tline:
				sline, tline = sline.decode("utf-8"), tline.decode("utf-8")
				bratio = get_bi_ratio(len(sline.split()), len(tline.split()))
				if bratio > bmaxratio:
					bmaxratio = bratio
				bmeanratio += bratio
				oratio = get_bi_ratio(len(sline.replace("@@ ", "").split()), len(tline.replace("@@ ", "").split()))
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
