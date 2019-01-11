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
	with open(srcfs, "rb") as fs, open(srcft, "rb") as ft:
		for sline, tline in zip(fs, ft):
			sline, tline = sline.strip(), tline.strip()
			if sline and tline:
				sline, tline = sline.decode("utf-8"), tline.decode("utf-8")
				bratio = ratio_bilingual(len(sline.split()), len(tline.split()))
				if bratio > bmaxratio:
					bmaxratio = bratio
				oratio = ratio_bilingual(len(sline.replace("@@ ", "").split()), len(tline.replace("@@ ", "").split()))
				if oratio > omaxratio:
					omaxratio = oratio

	print("Maximum bpe bilingual ratio is: %f\nMaximum original bilingual ratio is: %f" % (bmaxratio, omaxratio))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
