"""
Setup configuration for the defect-free package.
"""
from setuptools import setup, find_packages

setup(
    name="defect-free",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.4.0'
    ],
    author="Otto Savola",
    description="A package for simulating atom rearrangement in optical lattices",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
)