"""
Setup script for ETCC precipitation merging package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="etcc-precipitation",
    version="1.0.0",
    author="Based on Wei et al. (2023)",
    author_email="your.email@example.com",
    description="Extended Triple Collocation for precipitation fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/etcc_precipitation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.6b0",
        ],
    },
    keywords="precipitation, triple collocation, data fusion, remote sensing, hydrology",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/etcc_precipitation/issues",
        "Source": "https://github.com/yourusername/etcc_precipitation",
        "Paper": "https://doi.org/10.1029/2023GL105120",
    },
)
