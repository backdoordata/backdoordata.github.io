---
title: "Visualizing Tennessee's Housing Market"
date: 2020-02-11
tags: [Data Analysis & Visualization]
breadcrumbs: true
header:
    image: "/images/nash-header.jpg"
excerpt: "Data Analysis & Visualization"

toc: true
toc_label: " Sights-to-See:"
toc_icon: "hiking"
---

```python
import quandl
import pandas as pd
import numpy as np
```

# Building The Dataset
To start off this project, I will be using Housing Price Index (HPI) data of Tennessee counties collected by the well-known real estate company Zillow.

## Querying Our Data
I begin by bringing in a table of the names of all 95 Tennessee counties and their corresponding Zillow county codes, which can be found here alongside the actual datasets: https://www.quandl.com/data/ZILLOW-Zillow-Real-Estate-Research/documentation.  
We need this table to seamlessly process our Quandl query. However, it was undetectable by the native Pandas web-scraper, so I used Excel to clean and format the data.


```python
county_Qcodes = pd.read_excel('/Users/DrewWoods/Desktop/Py_Project_1/quandl_county_codes.xlsx')

codes_array = pd.Series(county_Qcodes['CODE']).apply(str).array
names_array = pd.Series(county_Qcodes['AREA']).array

county_Qcodes.head()
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
      <th>AREA</th>
      <th>CODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Anderson</td>
      <td>455</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Bedford</td>
      <td>1567</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Benton</td>
      <td>1913</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Bledsoe</td>
      <td>1949</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Blount</td>
      <td>319</td>
    </tr>
  </tbody>
</table>
</div>



We can now run our query using a simple for loop, itterating through the array of county codes above. The second for loop is to rename the columns of our new dataframe using the names array (otherwise labeled 'ZILLOW/CO455_ZHVIAH', 'ZILLOW/CO1567_ZHVIAH', ...)


```python
my_api_key = str("VztcNTe_rfAxrDXvK1au")
```


```python
master_df = pd.DataFrame()

for i in codes_array[0:]:
    """ Populates master_df with individual county data from 95 Zillow pages """
    
    query = ['ZILLOW/CO' + i + '_ZHVIAH' for i in codes_array]
    working_df = quandl.get( query, authtoken= my_api_key )
    
    if master_df.empty:
        master_df = working_df
    else:
        master_df = master_df.join(working_df)

#properly renames columns
for i in range(95):
    master_df.rename(columns = {master_df.columns[i]:names_array[i]+'_County_HPI'}, inplace = True)
        
```
        
```python
master_df.head(5)
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
      <th>ZILLOW/CO455_ZHVIAH - Value</th>
      <th>ZILLOW/CO1567_ZHVIAH - Value</th>
      <th>ZILLOW/CO1913_ZHVIAH - Value</th>
      <th>ZILLOW/CO1949_ZHVIAH - Value</th>
      <th>ZILLOW/CO319_ZHVIAH - Value</th>
      <th>ZILLOW/CO377_ZHVIAH - Value</th>
      <th>ZILLOW/CO1607_ZHVIAH - Value</th>
      <th>ZILLOW/CO1021_ZHVIAH - Value</th>
      <th>ZILLOW/CO1745_ZHVIAH - Value</th>
      <th>ZILLOW/CO543_ZHVIAH - Value</th>
      <th>...</th>
      <th>ZILLOW/CO943_ZHVIAH - Value</th>
      <th>ZILLOW/CO932_ZHVIAH - Value</th>
      <th>ZILLOW/CO2438_ZHVIAH - Value</th>
      <th>ZILLOW/CO1615_ZHVIAH - Value</th>
      <th>ZILLOW/CO320_ZHVIAH - Value</th>
      <th>ZILLOW/CO1903_ZHVIAH - Value</th>
      <th>ZILLOW/CO1671_ZHVIAH - Value</th>
      <th>ZILLOW/CO1787_ZHVIAH - Value</th>
      <th>ZILLOW/CO226_ZHVIAH - Value</th>
      <th>ZILLOW/CO341_ZHVIAH - Value</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1996-04-30</td>
      <td>71800.0</td>
      <td>52700.0</td>
      <td>44100.0</td>
      <td>NaN</td>
      <td>91500.0</td>
      <td>68000.0</td>
      <td>42500.0</td>
      <td>48500.0</td>
      <td>43300.0</td>
      <td>46900.0</td>
      <td>...</td>
      <td>45700.0</td>
      <td>47800.0</td>
      <td>NaN</td>
      <td>50700.0</td>
      <td>68900.0</td>
      <td>29000.0</td>
      <td>48500.0</td>
      <td>41800.0</td>
      <td>165400.0</td>
      <td>109200.0</td>
    </tr>
    <tr>
      <td>1996-05-31</td>
      <td>72000.0</td>
      <td>52800.0</td>
      <td>44200.0</td>
      <td>NaN</td>
      <td>91600.0</td>
      <td>68800.0</td>
      <td>42700.0</td>
      <td>49000.0</td>
      <td>43500.0</td>
      <td>46900.0</td>
      <td>...</td>
      <td>45400.0</td>
      <td>48000.0</td>
      <td>NaN</td>
      <td>50800.0</td>
      <td>68900.0</td>
      <td>28600.0</td>
      <td>48300.0</td>
      <td>42000.0</td>
      <td>168300.0</td>
      <td>109600.0</td>
    </tr>
    <tr>
      <td>1996-06-30</td>
      <td>72100.0</td>
      <td>53100.0</td>
      <td>44200.0</td>
      <td>NaN</td>
      <td>91600.0</td>
      <td>69500.0</td>
      <td>43000.0</td>
      <td>49400.0</td>
      <td>43800.0</td>
      <td>46900.0</td>
      <td>...</td>
      <td>45200.0</td>
      <td>48400.0</td>
      <td>NaN</td>
      <td>51000.0</td>
      <td>68900.0</td>
      <td>28300.0</td>
      <td>48200.0</td>
      <td>42200.0</td>
      <td>170900.0</td>
      <td>110100.0</td>
    </tr>
    <tr>
      <td>1996-07-31</td>
      <td>72300.0</td>
      <td>53500.0</td>
      <td>44300.0</td>
      <td>NaN</td>
      <td>91300.0</td>
      <td>70100.0</td>
      <td>43200.0</td>
      <td>50000.0</td>
      <td>44200.0</td>
      <td>47000.0</td>
      <td>...</td>
      <td>45100.0</td>
      <td>48700.0</td>
      <td>NaN</td>
      <td>51200.0</td>
      <td>69200.0</td>
      <td>28000.0</td>
      <td>48000.0</td>
      <td>42500.0</td>
      <td>173100.0</td>
      <td>110700.0</td>
    </tr>
    <tr>
      <td>1996-08-31</td>
      <td>72500.0</td>
      <td>54100.0</td>
      <td>44500.0</td>
      <td>NaN</td>
      <td>91000.0</td>
      <td>70500.0</td>
      <td>43300.0</td>
      <td>50500.0</td>
      <td>44500.0</td>
      <td>47100.0</td>
      <td>...</td>
      <td>45000.0</td>
      <td>49100.0</td>
      <td>NaN</td>
      <td>51400.0</td>
      <td>69700.0</td>
      <td>27700.0</td>
      <td>47900.0</td>
      <td>42900.0</td>
      <td>174900.0</td>
      <td>111300.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 95 columns</p>
</div>



The code above actually gave us an error (which I opted to exclude for your viewing pleasure), but still processed our query.  
Apparently there are seven counties in Tennessee that Zillow does not collect data for. 
No worries though, these counties happen to be lower population rural counties, having little effect on the state as a whole.
We will take note of that here, drop them from master_df, and continue.


```python
missing_values = master_df.select_dtypes(exclude = float)
master_df = master_df.select_dtypes(include = float)
```

(for documentation purposes, the missing counties are Claiborne, Clay, Hancock, Lake, Moore, Perry, and Pickett County)

## Assessing NaN's
We still have some cleaning up to do before our dataset is usable. A few counties are missing the last six entires, so we will use their last observed HPI entry to fill in the NaN's. To do this, I used the pandas method df.fillna(method='ffill').  

Although we are only concerned with 2009 onwards, I used the following two functions to assess the few pre-2009 NaN's as well incase we decide to revisit the full dataset later.


```python
def Find_Most_Similar(county_name):
    """Finds and sorts remaining counties by difference in average HPI, prints top 5, and 
    plots the county (county_name) alongside the top 2 most similar"""
    
    Closest_County = []
    
    for i in df.columns:
        difference_i = (master_df[str(i)]-master_df[county_name]).mean(0)
        Closest_County = np.append(Closest_County, difference_i)
    
    Closest_County = pd.Series(data=Closest_County, index=master_df.columns.transpose())
    Closest_County = abs(Closest_County)
    Closest_County = Closest_County.drop(index = county_name).sort_values()
    
    print(Closest_County.head(5))
    return master_df[[county_name, Closest_County.index[0], Closest_County.index[1]]].plot()

def Replace_NANcounty(county_name, replacement_county):
    """Used in conjunction with Find_Most_Similar to replace NaN's with best fit"""
    master_df[county_name].fillna(master_df[replacement_county], inplace=True)
    print(master_df[county_name])
```


```python

```


```python

```


```python

```
