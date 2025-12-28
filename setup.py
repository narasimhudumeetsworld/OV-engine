from setuptools import setup, find_packages

setup(
    name="openvinayaka",
    version="1.0.0",
    author="Prayaga Vaibhav (Akka)",
    description="Universal Hallucination-Free AI Engine",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "sentence-transformers",
        "numpy",
        "accelerate"
    ],
    entry_points={
        "console_scripts": [
            "openvinayaka=openvinayaka.cli:main"
        ]
    }
)
