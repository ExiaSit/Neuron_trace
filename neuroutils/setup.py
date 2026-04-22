from setuptools import setup, find_packages

setup(
    name="neuroutils",
    version="0.1.0",
    description="Utility library for neuron segmentation, reconstruction, and visualization.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-image",
        "scipy"
    ],
    python_requires=">=3.7",
)
