#encoding: utf-8

from utils.fmt.base import line_char_reader as file_reader
from utils.fmt.single import batch_padder as batch_padder_base

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, file_reader=file_reader, **kwargs):

	return batch_padder_base(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, file_reader=file_reader, **kwargs)
