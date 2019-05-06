#encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

from os import walk
from os.path import join as pjoin

def get_name(fname):

	return fname[:fname.rfind(".")].replace("/", ".")

def walk_path(ptw, eccargs):

	rsl = []
	for root, dirs, files in walk(ptw):
		for pyf in files:
			if pyf.endswith(".py"):
				_pyf = pjoin(root, pyf)
				if _pyf.find("/__") < 0:
					rsl.append(Extension(get_name(_pyf), [_pyf], extra_compile_args=eccargs))

	return rsl

if __name__ == "__main__":

	eccargs = ["-Ofast", "-march=native", "-pipe", "-fomit-frame-pointer"]

	baselist = ["loss.py", "lrsch.py", "utils.py", "translator.py", "discriminator.py"]

	extlist = [Extension(get_name(pyf), [pyf], extra_compile_args=eccargs) for pyf in baselist]

	for _mp in ("modules/", "parallel/", "transformer/"):
		extlist.extend(walk_path(_mp, eccargs))

	setup(cmdclass = {"build_ext": build_ext}, ext_modules=cythonize(extlist, quiet = True, language_level = 3))

