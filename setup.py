

import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent

requirements = [
    line.strip() for line in (here / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="SWEET_python",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
