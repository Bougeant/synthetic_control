---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
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
from datetime import datetime
import pandas as pd
from synthetic_control import SyntheticControl
```

# Read California proposition 99 dataset

```python
df = pd.read_csv("./california_prop99.csv")
```

```python
df["Year"] = pd.to_datetime(df["Year"], format='%Y')
```

```python
df = df.pivot(index="Year", columns="State",values="PacksPerCapita").round(1)
```

```python
df.head()
```

# Generate predictions for the Synthetic Control group

```python
sc = SyntheticControl(
    treatment_start=datetime(year=1989, month=1, day=1), 
    treatment_name="California",
)
```

```python
y_pred = sc.get_results(df)
```

### Composition of synthetic group

```python

```

# Display predictions for the Synthetic Control group


### Comparison

```python
sc.compare(df, y_pred, y_axis="Packs of cigarettes per Capita")
```

### Impact

```python
sc.impact(df, y_pred, y_axis="Packs of cigarettes per Capita")
```

# Testing for false positives


### Using a different treatment date

```python
sc_1992 = SyntheticControl(
    treatment_start=datetime(year=1992, month=1, day=1), 
    treatment_name="California",
)
```

```python
y_pred = sc_1992.get_results(df)
```

```python
sc_1992.compare(df, y_pred, y_axis="Packs of cigarettes per Capita")
```

### Application to the state of Pennsylvania 

```python
sc_penn = SyntheticControl(
    treatment_start=datetime(year=1989, month=1, day=1), 
    treatment_name="Pennsylvania",
)
```

```python
y_pred = sc_penn.get_results(df)
```

```python
sc_penn.compare(df, y_pred, y_axis="Packs of cigarettes per Capita")
```

```python

```
