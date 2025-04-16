```python
import pandas as pd
```

## Introduction

#### This dataset is centered around World Happiness Score from 2019


```python
df = pd.read_csv("C:/Users/madb3/Downloads/Dataset 19.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This data table shows rank, country, score, GDP per capita (), social support, life expectancy score(), freedom (), generosity, and perception of government corruption.
```


```python
# I would like to explore what contributes to high happiness score and where the U.S ranks in this list
```

## Data Cleaning


```python
#First make sure there are no null values
```


```python
print(df.isnull().sum())
```

    Overall rank                    0
    Country or region               0
    Score                           0
    GDP per capita                  0
    Social support                  0
    Healthy life expectancy         0
    Freedom to make life choices    0
    Generosity                      0
    Perceptions of corruption       0
    dtype: int64
    


```python
#See what kind of data is in the table
```


```python
df.dtypes
```




    Overall rank                      int64
    Country or region                object
    Score                           float64
    GDP per capita                  float64
    Social support                  float64
    Healthy life expectancy         float64
    Freedom to make life choices    float64
    Generosity                      float64
    Perceptions of corruption       float64
    dtype: object



## Visualization


```python
import seaborn as sns
import matplotlib.pyplot as plt
```

#### Before visualizing, I predict that social support and decision freedom will have the most correlation to happiness. Let's compare how each column contributes to happiness score

#### GDP Correlation


```python
factor = ['GDP per capita']
for f in factor:
    plt.figure(figsize=(6,4))
    plt.scatter(df[factor], df['Score'], alpha=0.6)
    plt.title('Happiness Score vs GDP')
    plt.xlabel(f)
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.show()
```


    
![png](output_15_0.png)
    


##### GDP has a steady poisitive correlation with Happiness score

#### Social Support Correlation


```python
factor = ['Social support']
for f in factor:
    plt.figure(figsize=(6,4))
    plt.scatter(df[factor], df['Score'], alpha=0.6)
    plt.title('Happiness Score vs Social Support')
    plt.xlabel(f)
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.show()
```


    
![png](output_18_0.png)
    


##### Social support appears to have a very strong positive correlation with happiness score

#### Life expectancy correlation


```python
factor = ['Healthy life expectancy']
for f in factor:
    plt.figure(figsize=(6,4))
    plt.scatter(df[factor], df['Score'], alpha=0.6)
    plt.title('Happiness Score vs Healthy Life Expectancy')
    plt.xlabel(f)
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.show()
```


    
![png](output_21_0.png)
    


##### There is a strong positive correlation between happiness score and healthy life expectancy

### Freedom Correlation


```python
factor = ['Freedom to make life choices']
for f in factor:
    plt.figure(figsize=(6,4))
    plt.scatter(df[factor], df['Score'], alpha=0.6)
    plt.title('Happiness Score vs Decision Freedom')
    plt.xlabel(f)
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.show()
```


    
![png](output_24_0.png)
    


##### There is a positive correlation between happiness score and decision freedom

#### Generosity Correlation


```python
factor = ['Generosity']
for f in factor:
    plt.figure(figsize=(6,4))
    plt.scatter(df[factor], df['Score'], alpha=0.6)
    plt.title('Happiness Score vs Generosity')
    plt.xlabel(f)
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.show()
```


    
![png](output_27_0.png)
    


##### Generosity has very little correlation to happiness score! We may want to drop this from the table

#### Negative Government Perception Correlation


```python
factor = ['Perceptions of corruption']
for f in factor:
    plt.figure(figsize=(6,4))
    plt.scatter(df[factor], df['Score'], alpha=0.6)
    plt.title('Happiness Score vs Negative Government Perception')
    plt.xlabel(f)
    plt.ylabel('Happiness Score')
    plt.grid(True)
    plt.show()
```


    
![png](output_30_0.png)
    



```python
ss_correlation = df['Score'].corr(df['Social support'])
print("Correlation coefficient: ", ss_correlation)
```

    Correlation coefficient:  0.7770577880638643
    


```python
gdp_correlation = df['Score'].corr(df['GDP per capita'])
print("Correlation coefficient: ", gdp_correlation)
```

    Correlation coefficient:  0.7938828678781276
    


```python
f_correlation = df['Score'].corr(df['Freedom to make life choices'])
print("Correlation coefficient: ", f_correlation)
```

    Correlation coefficient:  0.5667418257199902
    


```python
#GDP has the highest correlation with happiness score
```


```python
ngp_correlation = df['Score'].corr(df['Perceptions of corruption'])
print("Correlation coefficient: ", ngp_correlation)
```

    Correlation coefficient:  0.3856130708664787
    

### How does the U.S compare on this list?


```python
usa = df[df['Country or region'] == 'United States']
print(usa)
```

        Overall rank Country or region  Score  GDP per capita  Social support  \
    18            19     United States  6.892           1.433           1.457   
    
        Healthy life expectancy  Freedom to make life choices  Generosity  \
    18                    0.874                         0.454        0.28   
    
        Perceptions of corruption  
    18                      0.128  
    

##### The USA is ranked 19

##### Score: 6.892, GDP per capita: 1.4333, Social Support: 1.457, Healthy life expectancy: .874, Decision Freedom: .454


```python
#When compared to the top 5 happiest countries, Scandinavia and the Netherlands, the US has a higher GDP but a lower social support, healthy expectancy, and decision freedom score
```


```python
#Side by side bar chart to visualize differences
```


```python
import numpy as np
```


```python
subset = df[df['Country or region'].isin(['United States', 'Finland'])]
subset = subset.set_index('Country or region')
```


```python
factors = ['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices']

usa_results = subset.loc['United States', factors]
finland_results = subset.loc['Finland', factors]
```


```python
x = np.arange(len(factors))
```


```python
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, usa_results, width, label='USA', color='red')
plt.bar(x + width/2, finland_results, width, label='Finland', color='darkblue')
plt.xticks(x, factors, rotation=45, ha='right')
plt.ylabel('Score Contribution')
plt.title('USA vs Finland – Happiness Factor Comparison')
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](output_46_0.png)
    



```python
#I would like to see how the lowest ranks to the highest
```


```python
lowest_rank = df.nlargest(1, 'Overall rank')
print(lowest_rank)
```

         Overall rank Country or region  Score  GDP per capita  Social support  \
    155           156       South Sudan  2.853           0.306           0.575   
    
         Healthy life expectancy  Freedom to make life choices  Generosity  \
    155                    0.295                          0.01       0.202   
    
         Perceptions of corruption  
    155                      0.091  
    


```python
#South Sudan is the lowest ranked country from the dataset
```


```python
dr = df[df['Country or region'] == 'Dominican Republic']
print(dr)
```

        Overall rank   Country or region  Score  GDP per capita  Social support  \
    76            77  Dominican Republic  5.425           1.015           1.401   
    
        Healthy life expectancy  Freedom to make life choices  Generosity  \
    76                    0.779                         0.497       0.113   
    
        Perceptions of corruption  
    76                      0.101  
    


```python
lowest_rank = df.nlargest(5, 'Overall rank')
print(lowest_rank)
```

         Overall rank         Country or region  Score  GDP per capita  \
    155           156               South Sudan  2.853           0.306   
    154           155  Central African Republic  3.083           0.026   
    153           154               Afghanistan  3.203           0.350   
    152           153                  Tanzania  3.231           0.476   
    151           152                    Rwanda  3.334           0.359   
    
         Social support  Healthy life expectancy  Freedom to make life choices  \
    155           0.575                    0.295                         0.010   
    154           0.000                    0.105                         0.225   
    153           0.517                    0.361                         0.000   
    152           0.885                    0.499                         0.417   
    151           0.711                    0.614                         0.555   
    
         Generosity  Perceptions of corruption  
    155       0.202                      0.091  
    154       0.235                      0.035  
    153       0.158                      0.025  
    152       0.276                      0.147  
    151       0.217                      0.411  
    


```python
#The 5 least happy countries were all actively in wars and conflicts in 2019 with the exception of Tanzania. 
```
