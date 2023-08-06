from setuptools import setup, find_packages
import os

VERSION = '1.0.2' 
DESCRIPTION = 'Tensorflow ML'
LONG_DESCRIPTION = 'An abstract implementation of commonly used machine learning algorithms using TensorFlow 2.0'

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = [] # Here we'll add: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="tensorflow_ml", 
        version=VERSION,
        author="Siddhant Pathak",
        author_email="siddhantpathak2@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=install_requires, # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'tensorflow', 'ml'],
        classifiers= [
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
        ]
)