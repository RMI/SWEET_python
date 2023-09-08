from setuptools import setup, find_packages

setup(
    name="SWEET_python",
    version="0.1",
    packages=['SWEET_python'],
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "pycountry",
        "scipy",
        "matplotlib"
    ],
)
