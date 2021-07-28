from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="poold",
    version="0.0.2",
    author="Genevieve Flaspohler and Lester Mackey",
    author_email="geflaspohler@gmail.com",
    description="Python library for Optimistic Online Learning under Delay",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geflaspohler/poold",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"": "poold"},
    packages=find_packages(where="poold"),
    install_requires=['pandas',
                      'numpy',                     
                      'matplotlib',                     
                      'seaborn',                     
                      'scipy',                     
                      ],
    python_requires=">=3.6",
)
