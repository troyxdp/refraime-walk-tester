# Usage: python compile.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [

    Extension("utils.cloud_api",  ["utils/cloud_api.py"]),

    Extension("start_process",  ["start_process.py"]),
    Extension("rtsp_stream",  ["rtsp_stream.py"]),

]

for e in ext_modules:
    e.cython_directives = {'language_level':"3"}
    
setup(
    name = 'Refraime Realtime Streamer',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)