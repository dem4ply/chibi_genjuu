# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="chibi_genjuu",
    version="1.0.0",
    description="",
    license="",
    author="",
    packages=find_packages(),
    install_requires=[ 'chibi', "elasticsearch_dsl==2.7.5" ],
    long_description=long_description,
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
    ]
)
