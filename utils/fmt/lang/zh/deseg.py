#encoding: utf-8

from re import compile

re_space = compile(r"(?<![a-zA-Z])\s(?![a-zA-Z])").sub
re_final_comma = compile("\.$").sub

deseg = lambda x: re_final_comma(u"\u3002", re_space("", x).replace(",", u"\uff0c"))
