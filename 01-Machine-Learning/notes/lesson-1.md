To work with dataframes, use pandas

```python
import pandas as pd

data_frame = pd.read_csv('CAMINHO_OU_LINK')
```

## Usefull methods

`
data_frame.describe()
`

```
count
mean 
std -> desvio padrão
min
q1
mediana
q3
max
```

Ways to analise numeric data:

histogram
boxplot
correlation
scatterplot

# Basics of model training

## Understand the possible features, you can use AI for it

# Strategies for missing data

Before using a strategy, check if isnt a business rule

Remotion
Lines with few absent fields
Lines with many absent fields

## Simple inputation

Set the mean or median for the missing value   
Advanced inputation

# Data standardization

# Data normalization

- StandardScaler
- MinMaxScaler 

# Encoding e Data Leakage 
