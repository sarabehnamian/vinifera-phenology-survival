"""
Setup script for vinifera_phenology package.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README_PACKAGE.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="vinifera-phenology",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Phenology analysis package for Vitis vinifera using interval-censored survival analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vinifera-phenology",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "openpyxl>=3.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "models": [
            "lifelines>=0.27.0",
        ],
        "plotting": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "vinifera-survival=vinifera_phenology.cli:main",
        ],
    },
)

