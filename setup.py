from setuptools import find_packages, setup

setup(
    name="clintrialfinder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.32.3",
        "pandas>=2.2.3",
        "numpy>=2.2.3",
        "python-dotenv>=1.0.1",
        "aiohttp>=3.11.12",
        "tqdm>=4.67.1",
        "openai",
        "loguru>=0.7.0",
    ],
    author="Wang Qiansheng",
    author_email="chncwang@gmail.com",
    description="A package for crawling and analyzing clinical trials",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chncwang/ClinTrialFinder",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
