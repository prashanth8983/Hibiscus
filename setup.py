"""
Setup script for the Professional Production-Grade Transformer package.

This package provides a comprehensive, production-ready implementation
of the Transformer architecture with modern best practices.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [line.strip() for line in requirements_path.read_text().splitlines() 
                   if line.strip() and not line.startswith("#")]

setup(
    name="hibiscus-transformer",
    version="1.0.0",
    author="Hibiscus Team",
    author_email="team@hibiscus.ai",
    description="Professional Production-Grade Transformer Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hibiscus/transformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "transformer-train=scripts.train:main",
            "transformer-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "transformer": ["*.yaml", "*.yml"],
    },
    keywords=[
        "transformer",
        "attention",
        "neural-network",
        "deep-learning",
        "nlp",
        "machine-learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/hibiscus/transformer/issues",
        "Source": "https://github.com/hibiscus/transformer",
        "Documentation": "https://hibiscus-transformer.readthedocs.io/",
    },
) 