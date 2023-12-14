---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
cd ..
```

```python
from datetime import datetime
import pandas as pd
from synthetic_control import SyntheticControl
```

# Read California proposition 99 dataset

```python
df = pd.read_csv("./examples/california_prop99.csv")
```

```python
df["Year"] = pd.to_datetime(df["Year"], format='%Y')
```

```python
df = df.pivot(index="Year", columns="State",values="PacksPerCapita")
```

```python
X = df.drop(columns="California")
y = df["California"]
```

# Generate predictions for the Synthetic Control group

```python
sc = SyntheticControl(
    treatment_start=datetime(year=1989, month=1, day=1), 
    treatment_name="California",
    fit_intercept=True, 
    alpha=100, 
    ci_fraction=0.5
)
```

```python
y_ci = sc.get_confidence_interval(X, y)
```

### Composition of synthetic group

```python

```

# Display predictions for the Synthetic Control group


### Comparison

```python
sc.compare(y, y_ci, y_axis="Packs of cigarettes per Capita")
```

### Impact

```python

```

# Testing for false positives


### Using a different treatment date

```python
sc_1995 = SyntheticControl(
    treatment_start=datetime(year=1995, month=1, day=1), 
    treatment_name="California",
    fit_intercept=True, 
    alpha=100, 
    ci_fraction=0.5
)
```

```python
y_ci = sc_1995.get_confidence_interval(X, y)
```

```python
sc_1995.compare(y, y_ci, y_axis="Packs of cigarettes per Capita")
```

### Application to the state of Pennsylvania 

```python
X_penn = df.drop(columns="Pennsylvania")
y_penn = df["Pennsylvania"]
```

```python
sc_penn = SyntheticControl(
    treatment_start=datetime(year=1989, month=1, day=1), 
    treatment_name="Pennsylvania",
    fit_intercept=True, 
    alpha=100, 
    ci_fraction=0.5
)
```

```python
y_ci_penn = sc_penn.get_confidence_interval(X_penn, y_penn)
```

```python
sc_penn.compare(y_penn, y_ci_penn, y_axis="Packs of cigarettes per Capita")
```

```python

```
