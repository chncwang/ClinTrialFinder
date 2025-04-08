from setuptools import find_packages, setup

setup(
    name="clinical_trial_crawler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "pandas",
        "numpy",
        "python-dotenv",
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
