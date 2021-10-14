cd SRC_swig
swig -python LKH.i
python3 setup.py build_ext
cd ..
cp SRC_swig/build/lib.*/_LKH.*.so ./
