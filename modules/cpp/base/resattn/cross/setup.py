#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="res_cross_attn_cpp", ext_modules=[cpp_extension.CppExtension("res_cross_attn_cpp", ["modules/cpp/base/resattn/cross/attn.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
