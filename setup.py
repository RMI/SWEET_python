from setuptools import setup, find_packages

setup(
    name="SWEET_python",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        numpy
        pandas
        pycountry
        scipy
        matplotlib
    ],
)
