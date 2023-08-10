#encoding: utf-8

from html import unescape
from unicodedata import normalize as uni_norm_func

#"NFC", "NFD", "NFKD"
uni_normer = "NFKC"

def clean_sp_char(istr):

	rs = []
	for c in istr:
		num = ord(c)
		if num == 12288:
			rs.append(" ")
		elif (num > 65280) and (num < 65375):
			rs.append(chr(num - 65248))
		elif not ((num < 32 and num != 9) or (num > 126 and num < 161) or (num > 8202 and num < 8206) or (num > 57343 and num < 63744) or (num > 64975 and num < 65008) or (num > 65519)):
			rs.append(c)

	return "".join(rs)

def norm_u8_str(x, uni_normer=uni_normer):

	return unescape(clean_sp_char(uni_norm_func(uni_normer, x)))

def norm_u8_byte(x, uni_normer=uni_normer):

	return unescape(clean_sp_char(uni_norm_func(uni_normer, x.decode("utf-8")))).encode("utf-8")

def norm_u8(x, uni_normer=uni_normer):

	return norm_u8_byte(x, uni_normer=uni_normer) if isinstance(x, bytes) else norm_u8_str(x, uni_normer=uni_normer)
