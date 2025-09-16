from setuptools import setup, find_packages
import pathlib

# Read the contents of README.md
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="rag-doc-qa",
    version="1.0.0",
    description="Production-ready Retrieval-Augmented Generation system for document QA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    maintainer="Your Name",
    maintainer_email="your.email@example.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic"
    ],
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "rag_doc_qa": ["*.json", "*.yaml", "*.yml"],
    },
    install_requires=[],
    extras_require={},
    entry_points={
        "console_scripts": [
            "rag-qa=rag_doc_qa.web_app:main",
            "rag-eval=rag_doc_qa.evaluate:main",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/yourusername/rag-doc-qa",
        "Documentation": "https://github.com/yourusername/rag-doc-qa#readme",
        "Repository": "https://github.com/yourusername/rag-doc-qa",
        "Issues": "https://github.com/yourusername/rag-doc-qa/issues",
    },
)
