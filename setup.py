from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
      name='bioCaption',  # Required
      version='1.1.0',
      description='Diagnostic Captioning',
      url='https://github.com/nlpaueb/dc',
      author='nlpaueb',
      classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.7'
      ],
      keywords='diagnostic image captioning machine learning',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=['bs4'],
)
