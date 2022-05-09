import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autokoopman",
    version="0.1b",
    # TODO: change this
    author="Ethan Lew",
    author_email="ethanlew16@gmail.com",
    description="Koopman Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EthanJamesLew/AutoKoopman",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": ".", "autokoopman": "autokoopman"},
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.4",
        "scipy>=1.5.4",
        "sympy >= 1.9",
        "pandas >= 1.1.5",
        "scikit-learn >= 1.0.2" "tqdm>=4.62.2",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)
