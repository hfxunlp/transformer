#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="act_cpp", ext_modules=[cpp_extension.CppExtension("act_cpp", ["modules/cpp/act/act.cpp", "modules/cpp/act/act_func.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
