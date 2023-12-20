from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="financialdata",  # Required
    version="1.0.1",  # Required
    packages=find_packages(),
    description="financial mining",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  
    # Optional (see note above)
    url="https://github.com/R09722akaBennett/",  # Optional
    author="bennett",  # Optional
    author_email="samlin266118@gmail.com",  # Optional
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="financial, python",  # Optional
    project_urls={  # Optional
        "documentation": "https://linsamtw.github.io/FinMindDoc/",
        "Source": "https://github.com/R09722akaBennett/Side_Project/",
    },
)
