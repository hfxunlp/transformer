#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="pff_cpp", ext_modules=[cpp_extension.CppExtension("pff_cpp", ["modules/cpp/base/ffn/pff.cpp", "modules/cpp/base/ffn/pff_func.cpp", "modules/cpp/act/act_func.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
