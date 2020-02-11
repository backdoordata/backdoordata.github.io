```python
import quandl
import pandas as pd
import numpy as np
```

# Building the Dataset
To start off this project, I will be using Housing Price Index (HPI) data of Tennessee counties collected by the well-known real estate company Zillow.

## Querying
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
        
master_df.head(5)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-7-9e2fd5917697> in <module>
         10         master_df = working_df
         11     else:
    ---> 12         master_df = master_df.join(working_df)
         13 
         14 #properly renames columns


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in join(self, other, on, how, lsuffix, rsuffix, sort)
       7244         # For SparseDataFrame's benefit
       7245         return self._join_compat(
    -> 7246             other, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort
       7247         )
       7248 


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in _join_compat(self, other, on, how, lsuffix, rsuffix, sort)
       7267                 right_index=True,
       7268                 suffixes=(lsuffix, rsuffix),
    -> 7269                 sort=sort,
       7270             )
       7271         else:


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
         81         validate=validate,
         82     )
    ---> 83     return op.get_result()
         84 
         85 


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/reshape/merge.py in get_result(self)
        646 
        647         llabels, rlabels = _items_overlap_with_suffix(
    --> 648             ldata.items, lsuf, rdata.items, rsuf
        649         )
        650 


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/reshape/merge.py in _items_overlap_with_suffix(left, lsuffix, right, rsuffix)
       2009         raise ValueError(
       2010             "columns overlap but no suffix specified: "
    -> 2011             "{rename}".format(rename=to_rename)
       2012         )
       2013 


    ValueError: columns overlap but no suffix specified: Index(['ZILLOW/CO455_ZHVIAH - Value', 'ZILLOW/CO1567_ZHVIAH - Value',
           'ZILLOW/CO1913_ZHVIAH - Value', 'ZILLOW/CO1949_ZHVIAH - Value',
           'ZILLOW/CO319_ZHVIAH - Value', 'ZILLOW/CO377_ZHVIAH - Value',
           'ZILLOW/CO1607_ZHVIAH - Value', 'ZILLOW/CO1021_ZHVIAH - Value',
           'ZILLOW/CO1745_ZHVIAH - Value', 'ZILLOW/CO543_ZHVIAH - Value',
           'ZILLOW/CO701_ZHVIAH - Value', 'ZILLOW/CO964_ZHVIAH - Value',
           'ZILLOW/CO1712_ZHVIAH - Not Found', 'ZILLOW/CO2405_ZHVIAH - Not Found',
           'ZILLOW/CO1659_ZHVIAH - Value', 'ZILLOW/CO1514_ZHVIAH - Value',
           'ZILLOW/CO2295_ZHVIAH - Value', 'ZILLOW/CO1495_ZHVIAH - Value',
           'ZILLOW/CO69_ZHVIAH - Value', 'ZILLOW/CO1955_ZHVIAH - Value',
           'ZILLOW/CO1879_ZHVIAH - Value', 'ZILLOW/CO597_ZHVIAH - Value',
           'ZILLOW/CO2173_ZHVIAH - Value', 'ZILLOW/CO707_ZHVIAH - Value',
           'ZILLOW/CO1890_ZHVIAH - Value', 'ZILLOW/CO1604_ZHVIAH - Value',
           'ZILLOW/CO1531_ZHVIAH - Value', 'ZILLOW/CO1734_ZHVIAH - Value',
           'ZILLOW/CO880_ZHVIAH - Value', 'ZILLOW/CO1447_ZHVIAH - Value',
           'ZILLOW/CO2305_ZHVIAH - Value', 'ZILLOW/CO510_ZHVIAH - Value',
           'ZILLOW/CO1209_ZHVIAH - Value', 'ZILLOW/CO3100_ZHVIAH - Not Found',
           'ZILLOW/CO2192_ZHVIAH - Value', 'ZILLOW/CO1780_ZHVIAH - Value',
           'ZILLOW/CO1492_ZHVIAH - Value', 'ZILLOW/CO2252_ZHVIAH - Value',
           'ZILLOW/CO1759_ZHVIAH - Value', 'ZILLOW/CO1710_ZHVIAH - Value',
           'ZILLOW/CO850_ZHVIAH - Value', 'ZILLOW/CO2395_ZHVIAH - Value',
           'ZILLOW/CO1881_ZHVIAH - Value', 'ZILLOW/CO2336_ZHVIAH - Value',
           'ZILLOW/CO582_ZHVIAH - Value', 'ZILLOW/CO1885_ZHVIAH - Value',
           'ZILLOW/CO111_ZHVIAH - Value', 'ZILLOW/CO2407_ZHVIAH - Not Found',
           'ZILLOW/CO2482_ZHVIAH - Value', 'ZILLOW/CO1598_ZHVIAH - Value',
           'ZILLOW/CO3094_ZHVIAH - Value', 'ZILLOW/CO1691_ZHVIAH - Value',
           'ZILLOW/CO608_ZHVIAH - Value', 'ZILLOW/CO890_ZHVIAH - Value',
           'ZILLOW/CO381_ZHVIAH - Value', 'ZILLOW/CO1752_ZHVIAH - Value',
           'ZILLOW/CO1725_ZHVIAH - Value', 'ZILLOW/CO1404_ZHVIAH - Value',
           'ZILLOW/CO1520_ZHVIAH - Value', 'ZILLOW/CO1779_ZHVIAH - Value',
           'ZILLOW/CO1956_ZHVIAH - Value', 'ZILLOW/CO1573_ZHVIAH - Value',
           'ZILLOW/CO1275_ZHVIAH - Value', 'ZILLOW/CO3089_ZHVIAH - Not Found',
           'ZILLOW/CO1837_ZHVIAH - Value', 'ZILLOW/CO1716_ZHVIAH - Value',
           'ZILLOW/CO1835_ZHVIAH - Value', 'ZILLOW/CO2403_ZHVIAH - Not Found',
           'ZILLOW/CO3098_ZHVIAH - Not Found', 'ZILLOW/CO967_ZHVIAH - Value',
           'ZILLOW/CO1432_ZHVIAH - Value', 'ZILLOW/CO1715_ZHVIAH - Value',
           'ZILLOW/CO1506_ZHVIAH - Value', 'ZILLOW/CO490_ZHVIAH - Value',
           'ZILLOW/CO168_ZHVIAH - Value', 'ZILLOW/CO2225_ZHVIAH - Value',
           'ZILLOW/CO1938_ZHVIAH - Value', 'ZILLOW/CO1381_ZHVIAH - Value',
           'ZILLOW/CO35_ZHVIAH - Value', 'ZILLOW/CO931_ZHVIAH - Value',
           'ZILLOW/CO1946_ZHVIAH - Value', 'ZILLOW/CO1286_ZHVIAH - Value',
           'ZILLOW/CO257_ZHVIAH - Value', 'ZILLOW/CO520_ZHVIAH - Value',
           'ZILLOW/CO1099_ZHVIAH - Value', 'ZILLOW/CO943_ZHVIAH - Value',
           'ZILLOW/CO932_ZHVIAH - Value', 'ZILLOW/CO2438_ZHVIAH - Value',
           'ZILLOW/CO1615_ZHVIAH - Value', 'ZILLOW/CO320_ZHVIAH - Value',
           'ZILLOW/CO1903_ZHVIAH - Value', 'ZILLOW/CO1671_ZHVIAH - Value',
           'ZILLOW/CO1787_ZHVIAH - Value', 'ZILLOW/CO226_ZHVIAH - Value',
           'ZILLOW/CO341_ZHVIAH - Value'],
          dtype='object')



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
