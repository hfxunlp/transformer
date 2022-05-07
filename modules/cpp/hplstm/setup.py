#encoding: utf-8

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="lgate_cpp", ext_modules=[cpp_extension.CppExtension("lgate_cpp", ["modules/cpp/hplstm/lgate.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
