#encoding: utf-8

import sys

def handle(srcf, rsf, el1=10, el2=9, el3=7):

	acc_bsize = 0
	odd_line = True
	l1, l2, l3 = [], [], []
	ens = "\n".encode("utf-8")
	with open(srcf, "rb") as frd, open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8").split()
				if tmp[0][0].isalpha():
					if tmp[0] == "ES":
						if l1 and l2 and l3:
							l1, l2, l3 = l1[:el1], l2[:el2], l3[:el3]
							if min(l2) != l2[-1] and min(l3) != l3[-1]:
								fwrt.write(" & ".join(l1[:el1]).encode("utf-8"))
								fwrt.write(ens)
								fwrt.write(" & ".join(l2[:el2]).encode("utf-8"))
								fwrt.write(ens)
								fwrt.write(" & ".join(l3[:el3]).encode("utf-8"))
								fwrt.write(ens)
								fwrt.write(ens)
						l1, l2, l3 = [], [], []
						acc_bsize = 0
						odd_line = True
					else:
						if l1 and l2 and l3:
							l1, l2, l3 = l1[:el1], l2[:el2], l3[:el3]
							if min(l2) != l2[-1] and min(l3) != l3[-1]:
								fwrt.write(" & ".join(l1[:el1]).encode("utf-8"))
								fwrt.write(ens)
								fwrt.write(" & ".join(l2[:el2]).encode("utf-8"))
								fwrt.write(ens)
								fwrt.write(" & ".join(l3[:el3]).encode("utf-8"))
								fwrt.write(ens)
								fwrt.write(ens)
							l1, l2, l3 = [], [], []
							acc_bsize = 0
							odd_line = True
						_cur_bsize = int(tmp[-1])
						acc_bsize += _cur_bsize
				else:
					_cur_bsize = int(tmp[0])
					acc_bsize += _cur_bsize
				if odd_line:
					odd_line = False
				else:
					l1.append(str(acc_bsize))
					_n_ele = len(tmp)
					if _n_ele >= 3:
						l2.append(tmp[2])
					if _n_ele >= 7:
						l3.append(tmp[6])
					odd_line = True

		if l1 and l2 and l3:
			l1, l2, l3 = l1[:el1], l2[:el2], l3[:el3]
			if min(l2) != l2[-1] and min(l3) != l3[-1]:
				fwrt.write(" & ".join(l1[:el1]).encode("utf-8"))
				fwrt.write(ens)
				fwrt.write(" & ".join(l2[:el2]).encode("utf-8"))
				fwrt.write(ens)
				fwrt.write(" & ".join(l3[:el3]).encode("utf-8"))
				fwrt.write(ens)
				fwrt.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
