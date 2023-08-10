#encoding: utf-8

from Cython.Build import cythonize
from Cython.Compiler import Options
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
from os import walk
from os.path import join as pjoin

Options.docstrings = False
Options.embed_pos_in_docstring = False
Options.annotate = False

def get_name(fname):

	return fname[:fname.rfind(".")].replace("/", ".")

def legal(pname, fbl):

	rs = True
	for pyst, pyf in fbl:
		if pname.startswith(pyst) or pname == pyf:
			rs = False
			break

	return rs

def walk_path(ptw, eccargs):

	fbl = []
	tmpl = []
	for root, dirs, files in walk(ptw):
		for pyf in files:
			if pyf.endswith(".py") and (pyf != "setup.py"):
				_pyf = pjoin(root, pyf)
				if _pyf.find("/__") < 0:
					tmpl.append(_pyf)
			elif pyf.endswith(".nocy"):
				_pyst = pjoin(root, pyf)[:-4].replace("/", ".")
				_pyf = _pyst[:-1]
				fbl.append((_pyst, _pyf))

	rsl = []
	for pyf in tmpl:
		_pyc = get_name(pyf)
		if legal(_pyc, fbl):
			rsl.append(Extension(_pyc, [pyf], extra_compile_args=eccargs))

	return rsl

if __name__ == "__main__":

	eccargs = ["-Ofast", "-march=native", "-pipe", "-fomit-frame-pointer"]

	baselist = ["lrsch.py"]
	extlist = [Extension(get_name(pyf), [pyf], extra_compile_args=eccargs) for pyf in baselist]
	for _mp in ("parallel/", "loss/", "optm/", "modules/", "transformer/", "utils/", "datautils/",):
		_tmp = walk_path(_mp, eccargs)
		if _tmp:
			extlist.extend(_tmp)

	if extlist:
		setup(cmdclass = {"build_ext": build_ext}, ext_modules=cythonize(extlist, quiet=True, language_level=3))
