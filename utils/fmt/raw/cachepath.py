#encoding: utf-8

from uuid import uuid4 as uuid_func

from utils.base import mkdir

cache_file_prefix = "train"

def get_cache_path(*fnames):

	_cache_path = None
	for _t in fnames:
		_ = _t.rfind("/") + 1
		if _ > 0:
			_cache_path = _t[:_]
			break
	_uuid = uuid_func().hex
	if _cache_path is None:
		_cache_path = "cache/floader/%s/" % _uuid
	else:
		_cache_path = "%sfloader/%s/" % (_cache_path, _uuid,)
	mkdir(_cache_path)

	return _cache_path

def get_cache_fname(fpath, i=0, fprefix=cache_file_prefix):

	return "%s%s.%d.h5" % (fpath, fprefix, i,)
