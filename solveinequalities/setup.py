from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "interval",
        ["interval.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    )
]

setup(
    name="interval",
    ext_modules=ext_modules,
)
