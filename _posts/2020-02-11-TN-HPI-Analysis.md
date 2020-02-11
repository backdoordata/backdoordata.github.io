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

# Building The Dataset
To start off this project, I will be using Housing Price Index (HPI) data of Tennessee counties collected by the well-known real estate company Zillow.

## Querying Our Data
I begin by bringing in a table of the names of all 95 Tennessee counties and their corresponding Zillow county codes, which can be found here alongside the actual datasets: https://www.quandl.com/data/ZILLOW-Zillow-Real-Estate-Research/documentation.  
We need this table to seamlessly process our Quandl query. However, it was undetectable by the native Pandas web-scraper, so I used Excel to clean and format the data.


```python
import quandl
import pandas as pd
import numpy as np

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
df = pd.read_csv("/Users/DrewWoods/Desktop/Py_Project_1/TNCounty_HPIs2.csv", index_col= 'Date')
df
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
      <th>Anderson_County_HPI</th>
      <th>Bedford_County_HPI</th>
      <th>Benton_County_HPI</th>
      <th>Bledsoe_County_HPI</th>
      <th>Blount_County_HPI</th>
      <th>Bradley_County_HPI</th>
      <th>Campbell_County_HPI</th>
      <th>Cannon_County_HPI</th>
      <th>Carroll_County_HPI</th>
      <th>Carter_County_HPI</th>
      <th>...</th>
      <th>Unicoi_County_HPI</th>
      <th>Union_County_HPI</th>
      <th>Van Buren_County_HPI</th>
      <th>Warren_County_HPI</th>
      <th>Washington_County_HPI</th>
      <th>Wayne_County_HPI</th>
      <th>Weakley_County_HPI</th>
      <th>White_County_HPI</th>
      <th>Williamson_County_HPI</th>
      <th>Wilson_County_HPI</th>
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
      <td>35500.0</td>
      <td>91500.0</td>
      <td>68000.0</td>
      <td>42500.0</td>
      <td>48500.0</td>
      <td>43300.0</td>
      <td>46900.0</td>
      <td>...</td>
      <td>45700.0</td>
      <td>47800.0</td>
      <td>26813.0</td>
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
      <td>34900.0</td>
      <td>91600.0</td>
      <td>68800.0</td>
      <td>42700.0</td>
      <td>49000.0</td>
      <td>43500.0</td>
      <td>46900.0</td>
      <td>...</td>
      <td>45400.0</td>
      <td>48000.0</td>
      <td>26995.0</td>
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
      <td>34200.0</td>
      <td>91600.0</td>
      <td>69500.0</td>
      <td>43000.0</td>
      <td>49400.0</td>
      <td>43800.0</td>
      <td>46900.0</td>
      <td>...</td>
      <td>45200.0</td>
      <td>48400.0</td>
      <td>27141.0</td>
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
      <td>33600.0</td>
      <td>91300.0</td>
      <td>70100.0</td>
      <td>43200.0</td>
      <td>50000.0</td>
      <td>44200.0</td>
      <td>47000.0</td>
      <td>...</td>
      <td>45100.0</td>
      <td>48700.0</td>
      <td>27391.0</td>
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
      <td>33100.0</td>
      <td>91000.0</td>
      <td>70500.0</td>
      <td>43300.0</td>
      <td>50500.0</td>
      <td>44500.0</td>
      <td>47100.0</td>
      <td>...</td>
      <td>45000.0</td>
      <td>49100.0</td>
      <td>27616.0</td>
      <td>51400.0</td>
      <td>69700.0</td>
      <td>27700.0</td>
      <td>47900.0</td>
      <td>42900.0</td>
      <td>174900.0</td>
      <td>111300.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2018-05-31</td>
      <td>134800.0</td>
      <td>140800.0</td>
      <td>76100.0</td>
      <td>87000.0</td>
      <td>176200.0</td>
      <td>141600.0</td>
      <td>95400.0</td>
      <td>137600.0</td>
      <td>69600.0</td>
      <td>109500.0</td>
      <td>...</td>
      <td>112300.0</td>
      <td>120300.0</td>
      <td>53200.0</td>
      <td>94300.0</td>
      <td>146600.0</td>
      <td>76300.0</td>
      <td>70900.0</td>
      <td>104400.0</td>
      <td>452700.0</td>
      <td>272900.0</td>
    </tr>
    <tr>
      <td>2018-06-30</td>
      <td>135700.0</td>
      <td>140300.0</td>
      <td>76900.0</td>
      <td>87700.0</td>
      <td>176000.0</td>
      <td>142900.0</td>
      <td>95300.0</td>
      <td>138300.0</td>
      <td>69600.0</td>
      <td>109700.0</td>
      <td>...</td>
      <td>112400.0</td>
      <td>120300.0</td>
      <td>53200.0</td>
      <td>94500.0</td>
      <td>147200.0</td>
      <td>76700.0</td>
      <td>70600.0</td>
      <td>105000.0</td>
      <td>453400.0</td>
      <td>273700.0</td>
    </tr>
    <tr>
      <td>2018-07-31</td>
      <td>136500.0</td>
      <td>141000.0</td>
      <td>76300.0</td>
      <td>88600.0</td>
      <td>176400.0</td>
      <td>144400.0</td>
      <td>95500.0</td>
      <td>139500.0</td>
      <td>69900.0</td>
      <td>110200.0</td>
      <td>...</td>
      <td>113100.0</td>
      <td>120800.0</td>
      <td>53200.0</td>
      <td>94900.0</td>
      <td>147300.0</td>
      <td>77600.0</td>
      <td>70700.0</td>
      <td>105800.0</td>
      <td>453700.0</td>
      <td>274800.0</td>
    </tr>
    <tr>
      <td>2018-08-31</td>
      <td>136800.0</td>
      <td>141600.0</td>
      <td>73800.0</td>
      <td>90400.0</td>
      <td>178100.0</td>
      <td>145800.0</td>
      <td>96300.0</td>
      <td>141800.0</td>
      <td>70800.0</td>
      <td>111600.0</td>
      <td>...</td>
      <td>114800.0</td>
      <td>122400.0</td>
      <td>53200.0</td>
      <td>96200.0</td>
      <td>147100.0</td>
      <td>78900.0</td>
      <td>70500.0</td>
      <td>107300.0</td>
      <td>454900.0</td>
      <td>276700.0</td>
    </tr>
    <tr>
      <td>2018-09-30</td>
      <td>137100.0</td>
      <td>142200.0</td>
      <td>71700.0</td>
      <td>92000.0</td>
      <td>179700.0</td>
      <td>146800.0</td>
      <td>97100.0</td>
      <td>143700.0</td>
      <td>71800.0</td>
      <td>113000.0</td>
      <td>...</td>
      <td>116400.0</td>
      <td>123800.0</td>
      <td>53200.0</td>
      <td>97500.0</td>
      <td>146600.0</td>
      <td>79900.0</td>
      <td>70000.0</td>
      <td>108600.0</td>
      <td>456100.0</td>
      <td>278500.0</td>
    </tr>
  </tbody>
</table>
<p>270 rows × 88 columns</p>
</div>




```python

```
