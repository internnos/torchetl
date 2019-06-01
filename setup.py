import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchetl",
    version="0.0.4",
    author="jedi",
    author_email="amajidsinar@gmail.com",
    description="Efficiently Extract, Transform, and Load your dataset into PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amajidsinar/torchetl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)