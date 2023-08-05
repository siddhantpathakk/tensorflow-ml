from setuptools import setup, find_packages

VERSION = '1.0.2' 
DESCRIPTION = 'Tensorflow ML'
LONG_DESCRIPTION = 'An abstract implementation of commonly used machine learning algorithms using TensorFlow 2.0'

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
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'tensorflow', 'ml'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)