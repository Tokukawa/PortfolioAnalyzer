import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="portfolio_analyzer",
    version="0.0.1",
    author="Tokukawa",
    author_email="emanuele.luzio@gmail.com",
    description="This package is a collection of tools I use to analyze portfolios.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/Tokukawa/PortfolioAnalyzer",
    install_requires=[
        "sklearn",
        "matplotlib",
        "pandas",
        "yahoofinancials",
        "cvxopt",
        "statsmodels  ",
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
