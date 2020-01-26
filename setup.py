from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
      name='dc',  # Required
      version='0.1',
      description='Diagnostic Captioning',
      url='https://github.com/nlpaueb/dc',
      author='nlpaueb',
      classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.7'
      ],
      keywords='diagnostic image captioning machine learning',
      package_dir={'': 'dc'},
      packages=find_packages(where='dc'),  # Required
      python_requires='>=2.7',
      install_requires=['bs4'],
)
