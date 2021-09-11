from setuptools import find_packages, setup

version = open("giants/__version__.py").read().strip('"\n')
long_description = open("docs/index.md", "r", encoding="utf-8").read()
required = open("requirements.txt", "r", encoding="utf-8").read().strip().split()

setup(
    name="giants",
    url="https://the.forestobservatory.com/giants",
    author="Salo Sciences",
    author_email="cba@salo.ai",
    license="MIT",
    description="Statistical and geospatial modeling tools for mapping big trees.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "ecology",
        "conservation",
        "remote sensing",
        "machine learning",
    ],
    packages=find_packages(exclude="tests"),
    version=version,
    python_requires=">=3.0",
    platforms="any",
    install_requires=required,
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
