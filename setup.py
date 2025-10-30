"""
Setup configuration for Collocation Analysis package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="collocation-analysis",
    version="1.0.0",
    author="Converted from MATLAB",
    author_email="licm_13@163.com",
    description="Comprehensive Python package for collocation-based error analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Collocation-Analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "examples": [
            "matplotlib>=3.1.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "collocation-example=examples.example_all_methods:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "collocation",
        "error-analysis",
        "remote-sensing",
        "data-validation",
        "triple-collocation",
        "data-fusion",
        "uncertainty-quantification",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Collocation-Analysis/issues",
        "Source": "https://github.com/yourusername/Collocation-Analysis",
        "Documentation": "https://github.com/yourusername/Collocation-Analysis/blob/main/README.md",
    },
)
