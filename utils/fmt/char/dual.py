#encoding: utf-8

from utils.fmt.base import line_char_reader as file_reader
from utils.fmt.dual import batch_padder as batch_padder_base

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, file_reader=file_reader, **kwargs):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, file_reader=file_reader, **kwargs)
