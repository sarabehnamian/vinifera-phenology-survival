"""
Setup script for the vinifera_phenology package.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README_PACKAGE.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="vinifera-phenology",
    version="0.1.0",
    author="Sara Behnamian",
    author_email="sara.behnamian@biol.lu.se",
    description=(
        "Interval-censored survival analysis of Vitis vinifera phenology "
        "with exogenous fixed pre-season weather windows"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarabehnamian/vinifera-phenology-survival",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",

    # Required to import and use the vinifera_phenology package itself
    # (survival_analysis, weather, utils, models, cli).
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "openpyxl>=3.0.0",   # pandas engine for .xlsx read/write
        "requests>=2.25.0",  # NASA POWER API calls in weather.py
    ],

    extras_require={
        # Survival model fitting (scripts 05, 07, 09, 10, 12, 13 and RE1_A1,
        # RE1_A3, RE1_A4, RE1_S2).
        "models": [
            "lifelines>=0.27.0",
        ],
        # Figure generation (scripts 04, 08, 09, 11-15 and most RE1_/RE2_ scripts).
        "plotting": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        # Statistical tests, diagnostics, and the hierarchical-dependence analysis
        # reported in the manuscript. scipy: RE1_A3, RE1_A4, RE1_A5, RE1_S2,
        # RE1_S3, RE1_S4. statsmodels: RE1_A5 (cluster-robust SEs).
        # scikit-learn: RE1_A2 (endogeneity simulation).
        "stats": [
            "scipy>=1.7.0",
            "statsmodels>=0.13.0",
            "scikit-learn>=1.0.0",
        ],
        # Everything needed to reproduce the analyses in the paper.
        "analysis": [
            "lifelines>=0.27.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "scipy>=1.7.0",
            "statsmodels>=0.13.0",
            "scikit-learn>=1.0.0",
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
