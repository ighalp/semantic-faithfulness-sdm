"""
Setup script for Semantic Divergence Metrics (SDM) package
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Filter out optional dependencies
core_requirements = [req for req in requirements if not any(
    optional in req for optional in ['openai', 'anthropic', 'python-dotenv', 'pytest', 'black', 'flake8', 'mypy']
)]

setup(
    name="semantic-faithfulness-sdm",
    version="1.0.0",
    author="Igor Halperin",
    author_email="your-email@example.com",
    description="Information-theoretic metrics for evaluating LLM faithfulness and semantic alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semantic-faithfulness-sdm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "python-dotenv>=0.19.0",
        ],
    },
    keywords="nlp, llm, faithfulness, information-theory, semantic-analysis, entropy, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/semantic-faithfulness-sdm/issues",
        "Source": "https://github.com/yourusername/semantic-faithfulness-sdm",
        "Documentation": "https://github.com/yourusername/semantic-faithfulness-sdm/docs",
    },
)
