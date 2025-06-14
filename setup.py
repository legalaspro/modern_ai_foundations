from setuptools import setup, find_packages

setup(
    name="modern_ai_foundations",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0",
        "torchvision",
        "transformers",
        "datasets",
        "matplotlib",
        "numpy",
        "scipy",
        "python-dotenv",
        "tqdm",
    ],
)