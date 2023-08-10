#encoding: utf-8

# portal from: https://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/deseg.py

import sys

from utils.fmt.base import sys_open
from utils.fmt.lang.zh.deseg import deseg as map_func

def handle(srcf, rsf):

	ens = "\n".encode("utf-8")
	with sys_open(srcf, "rb") as frd, sys_open(rsf, "wb") as fwrt:
		for _ in frd:
			fwrt.write(map_func(_.decode("utf-8").rstrip("\r\n")).encode("utf-8"))
			fwrt.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
