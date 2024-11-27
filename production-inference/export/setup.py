from distutils.core import setup
from distutils.extension import Extension 
from pathlib import Path
import os
from typing import List

from Cython.Build import cythonize

"""
This script cythonize the src folder of a project 

The script is considered to be placed near the src folder 
"""


IGNORED_FILES = {'entrypoint.py', '__init__.py'}
PACKAGE_DIR = Path('src')
BUILD_DIR = Path('build')

SRC_DIR = BUILD_DIR / 'src' # where .c files will be located
SRC_DIR.mkdir(parents=True, exist_ok=True)


def create_ext_modules(c_files: List[str]) -> List[Extension]:
    modules = []
    for c_f in c_files:
        filename, _ = os.path.splitext(c_f)
        ext_name = str(BUILD_DIR/filename).replace(os.path.sep, '.')
        extension = Extension(ext_name, sources=[str(SRC_DIR / c_f)])
        modules.append(extension)
    return modules

def cythonize_package_dir() -> List[str]:
    c_files = []
    for f in PACKAGE_DIR.glob('**/*'):
        if f.suffix != '.py' or f.name in IGNORED_FILES:
            continue
        stripped_path = os.path.relpath(str(f), str(PACKAGE_DIR))
        c_files.append(stripped_path.replace('.py', '.c'))
        _ = cythonize(str(f), build_dir=str(BUILD_DIR))
    return c_files

def main():
    print("3333333333333333333333333333333333333333333333333333333333333333333333333333")
    c_files = cythonize_package_dir()
    print(c_files)
    
    ext_modules = create_ext_modules(c_files)
    setup(name = 'PL_NN', ext_modules=ext_modules)
    return None

if __name__ == '__main__':
    main()