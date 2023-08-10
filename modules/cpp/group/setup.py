#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="group_cpp", ext_modules=[cpp_extension.CppExtension("group_cpp", ["modules/cpp/group/group.cpp", "modules/cpp/group/group_func.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
