import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autokoopman",
    version="0.21",
    description="Automated Koopman Operator Linearization Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EthanJamesLew/AutoKoopman",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    package_dir={"": ".", "autokoopman": "autokoopman"},
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.4",
        "scipy>=1.5.4",
        "sympy >= 1.9",
        "pandas >= 1.1.5",
        "scikit-learn >= 1.0.2",
        "tqdm>=4.62.2",
        "pysindy==1.7.2",
    ],
    extra_requires={
        'deepk': [
            "torch>=1.12.1",
        ],
        'bopt': [
            "gpy==1.10.0",
            "gpyopt==1.2.6",
        ],
    },
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)
