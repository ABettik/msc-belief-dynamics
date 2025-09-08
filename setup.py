from setuptools import setup, find_packages

setup(
    name='my_msc_proj',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
