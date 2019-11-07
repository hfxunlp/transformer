#encoding: utf-8

''' usage:
	python tools/add_tag.py $src_file.t7 $rs_file.t7 $token
'''

import sys

def handle(srcf, rsf, token):

	_et = token.encode("utf-8") if token.endswith(" ") else (token + " ").encode("utf-8")
	_ens = "\n".encode("utf-8")

	with open(srcf, "rb") as frd, open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				fwrt.write(_et)
				fwrt.write(tmp.encode("utf-8"))
				fwrt.write(_ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
