from setuptools import setup, find_packages

setup(
    name="DeepLearningNumpy",
    version="0.0.9",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.24.2"
    ]
)