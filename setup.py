

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
    # Decoration, not a version. SWEET never ships on its own -- it reaches the world
    # only inside a Climate TRACE run or a WasteMAP deploy, each of which has an
    # identity already -- so there is no hand-maintained number here to keep honest.
    # What a caller should read is SWEET_python.__version__, which is the commit this
    # copy was installed from. See the package docstring, and VERSIONING.md in
    # RMI_Climate_TRACE_Waste_Methane.
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
