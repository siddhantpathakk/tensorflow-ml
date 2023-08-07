from setuptools import setup, find_packages
import os

VERSION = '1.1.3' 

DESCRIPTION = 'Tensorflow ML'

with open("README.md", 'r') as f:
    LONG_DESCRIPTION = f.read()

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = [] 
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
        name="tensorflow-ml", 
        version=VERSION,
        author="Siddhant Pathak",
        author_email="siddhantpathak2@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",  # Specify the type of markup used in README.md
        packages=find_packages(),
        install_requires=install_requires,        
        keywords=['python', 'tensorflow', 'ml', 'keras'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
        ]
)
