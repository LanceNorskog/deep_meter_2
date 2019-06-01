import setuptools

# thanks to ceddlyburge/python_world sample

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Deep_Meter",
    version="0.0.1",
    author="Foo Bar",
    author_email="foo@bar.com",
    description="CMU Pronunciation Dictionary as code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LanceNorskog/deep_meter_2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
