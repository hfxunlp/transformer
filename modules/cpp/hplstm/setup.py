#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="lgate_cpp", ext_modules=[cpp_extension.CppExtension("lgate_cpp", ["modules/cpp/hplstm/lgate.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="lgate_nocx_cpp", ext_modules=[cpp_extension.CppExtension("lgate_nocx_cpp", ["modules/cpp/hplstm/lgate_nocx.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
