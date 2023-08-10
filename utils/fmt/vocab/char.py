#encoding: utf-8

from utils.fmt.base import list_reader_wst as file_reader
from utils.fmt.vocab.token import ldvocab as ldvocab_base, ldvocab_freq as ldvocab_freq_base, ldvocab_list as ldvocab_list_base, save_vocab as save_vocab_base

# use tab as seperator to keep the space token in the vocab
sep_load = sep_save = "\t"

def ldvocab(*args, sep=sep_load, file_reader=file_reader, **kwargs):

	return ldvocab_base(*args, sep=sep, file_reader=file_reader, **kwargs)

def save_vocab(*args, sep=sep_save, **kwargs):

	return save_vocab_base(*args, sep=sep, **kwargs)

def ldvocab_list(*args, sep=sep_load, file_reader=file_reader, **kwargs):

	return ldvocab_list_base(*args, sep=sep, file_reader=file_reader, **kwargs)

def ldvocab_freq(*args, sep=sep_load, file_reader=file_reader, **kwargs):

	return ldvocab_freq_base(*args, sep=sep, file_reader=file_reader, **kwargs)
