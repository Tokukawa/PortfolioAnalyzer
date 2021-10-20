# PortfolioAnalyzer

This package contains the tools I use to analyze stocks and portfolios. The package is pip installable. So you can use it typing

```bash
pip install git+https://github.com/Tokukawa/PortfolioAnalyzer.git
```

Take a look at the [jupyter examples](Examples/Overview.ipynb) to see what you can do with this package.

Namaste.


### Local installation

```bash
pip install .
```
Depends on your OS and arch (OSX mostly), maybe you will have this error:

> failed building wheel for scs

To fix that you need to pass explicit your arch:

```bash
ARCHFLAGS="-arch x86_64"  pip install .
```
