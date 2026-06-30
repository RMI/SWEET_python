

import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent

# Read the abstract (unpinned) dependencies from requirements.in so consumers are
# not over-constrained. The pinned lockfile in requirements.txt is generated from
# this file with pip-tools (pip-compile) and is used for reproducible installs and
# Dependabot scanning, not for install_requires.
requirements = [
    line.strip() for line in (here / "requirements.in").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="SWEET_python",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
