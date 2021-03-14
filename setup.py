from setuptools import setup, find_packages
# TODO setup.cfg with all metadata and dependencies list

setup(
    name='Topic Modeling Independent Study',
    author='Virginia Partridge',
    packages=find_packages(),
    long_description=open('README.md').read(),
    install_requires=[],
    tests_require=['pytest'],
)
