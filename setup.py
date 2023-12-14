from setuptools import find_packages, setup

requirements = ["pandas", "scikit-learn", "plotly"]

dev_requirements = [
    "black",
    "isort",
    "flake8",
    "bumpversion",
    "coverage",
    "pytest",
    "pytest-cov",
]

test_requirements = ["tox"]

jupyter_requirements = ["jupyter", "jupyterlab<4", "jupytext"]

setup(
    name="synthetic-control",
    version="0.0.1",
    url="http://github.com/Bougeant/synthetic_control",
    license_files=("LICENSE.txt",),
    description=("A user friendly package to use the Synthetic Control Method"),
    long_description=open("README.rst", "r").read(),
    author="Olivier Bougeant",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": test_requirements,
        "jupyter": jupyter_requirements,
    },
    entry_points={"console_scripts": []},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
