import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dc-nlpaueb",
    version="0.0.1",
    author="Vasiliki Kougia, Maria Georgiou, Ioannis Pavlopoulos",
    author_email="author@example.com",
    description="Diagnostic captioning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nlpaueb/dc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)