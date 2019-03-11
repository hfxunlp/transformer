#encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

from os import walk
from os.path import join as pjoin

def get_name(fname):

	return fname[:fname.rfind(".")].replace("/", ".")

if __name__ == "__main__":

	eccargs = ["-Ofast", "-march=native", "-pipe", "-fomit-frame-pointer"]

	baselist = ["TAmodules.py", "modules.py", "loss.py", "lrsch.py", "utils.py","rnncell.py", "translator.py", "discriminator.py"]

	extlist = [Extension(get_name(pyf), [pyf], extra_compile_args=eccargs) for pyf in baselist]

	for root, dirs, files in walk("parallel/"):
		for pyf in files:
			if pyf.endswith(".py"):
				_pyf = pjoin(root, pyf)
				if _pyf.find("/__") < 0:
					extlist.append(Extension(get_name(_pyf), [_pyf], extra_compile_args=eccargs))

	for root, dirs, files in walk("transformer/"):
		for pyf in files:
			if pyf.endswith(".py"):
				_pyf = pjoin(root, pyf)
				if _pyf.find("/__") < 0:
					extlist.append(Extension(get_name(_pyf), [_pyf], extra_compile_args=eccargs))

	setup(cmdclass = {"build_ext": build_ext}, ext_modules=cythonize(extlist, quiet = True, language_level = 3))

