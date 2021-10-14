from distutils.core import setup, Extension
import glob

FILES = glob.glob("*.c")

print (FILES)

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# FILES += ["LKH.i"]
# FILESN = []
# for f in FILES:
#     if f != "LKH_wrap.c":
#         FILESN.append(f)
# print (FILESN)
# FILESN=["LKH.i", "LKHmain.c"]
# FILESN=["LKHmain.c"]
ARGS = ['-O3', '-Wall', '-IINCLUDE', '-DTWO_LEVEL_TREE', '-g']
# ARGS = []

example_module = Extension('_LKH',
                           sources=FILES,
                           include_dirs=["INCLUDE", numpy_include],
                           extra_compile_args=ARGS,)

setup(name='LKH',
      version='0.1',
      author="SWIG Docs",
      description="""Simple swig example from docs""""",
      ext_modules=[example_module],
      py_modules=["LKH"],
      )
