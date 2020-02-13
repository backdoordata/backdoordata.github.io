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

One characteristic that's easily described by an area's HPI is how desirable it is to live there, aka the demand.  
Truncating the data to start at a more recent date is will magnify small, yet important, details since HPI data from 1996 isn't very indicative of upcoming/future HPI trends. Only as of lately can you begin to see noticable variation in the housing price index of different counties. More specifically, the standard deviation has increased 34% since 1996 (notice the "widening" of different HPI curves below).


```python
df.plot(title = 'Housing Price Index ($)' ,legend= False);
```


![png](/images/HPI_linegraph.png)


It comes to no surprise that the counties are positively correlated, but it's important to acknowledge this because it typically implies that we need to manipulate and/or normalize the data in order to see even small amounts of detail.  
  
The Great Recession 'officially' ended June of 2009, but as you can tell from the graph, the housing market continued to suffer. In the following lines of code, we truncate the dataframe using an approximation of when the housing market actually began to recover from the recession.


```python
df_09_plus = df.loc['2009-6-30':]
min_months = []

#creates list of post recession turning-point dates
for i in np.arange(0,88):
    itter_county = df_09_plus.iloc[:,i]
    local_min = min(itter_county)
    min_mo = df_09_plus.index.where(itter_county == local_min).dropna().values
    min_mo = pd.to_datetime(min_mo.item(0))
    min_months.append(min_mo)
    
#returns the most observed date from list 
#truncates df at that point
start_date = max(set(min_months), key=min_months.count)
df = df[df.index >= start_date]
```

Having this, May 2012, as the data's earliest entry is ideal.  
Why? By focusing on the housing market only during the recent economic upturn can show us a multitude of things, such as which areas are deemed 'most attractive' now that people are financially sound again, which markets are more stable and fluctuate the least, which areas are technologically/business progressive, which areas to invest in, etc..  
Lot's of these same characteristics can be seen in markets during the time leading up to the recession as well!  
  
We use percent change as a basis in which to evaluate these characteristics.  
The following code produces the necessary dataframe.


```python
#create new dataframe with entries as %change in HPI from May 2012
pct_func = lambda x : round(100*(x/(df.iloc[0].values) - 1),1)
df2 = pct_func(df.iloc[:])

#relabel columns of the new dataframe
new_cols = []
for i in df2.columns:
    pct_col = i.strip('_HPI')+'_pct'
    new_cols = np.append(new_cols, pct_col)
    
df2.columns = new_cols
df2
```
## Visualizing With Matplotib
To avoid another plot of 88 HPI curves, I took advantage of summary statistics for each of the monthly observations. Then using Matplotlib, I create a time series line plot describing the HPI mean growth and the first, second, and third standard deviations.


```python
DataFrame of summary statistics for our plot
d = {'mean':df2.transpose().mean(), 
     'std':df2.transpose().std(), 
     'mean+std':df2.transpose().mean()+df2.transpose().std(),
     'mean-std':df2.transpose().mean()-df2.transpose().std(),
     'mean+2std':df2.transpose().mean()+2*(df2.transpose().std()),
     'mean-2std':df2.transpose().mean()-2*(df2.transpose().std()),
     'mean+3std':df2.transpose().mean()+3*(df2.transpose().std()),
     'mean-3std':df2.transpose().mean()-3*(df2.transpose().std())
     }
df2_info = pd.DataFrame(index = df2.index, data= d)


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=[12.8, 9.6])
plt.margins(0,0)
plt.style.use('seaborn-darkgrid')

#mean curve
plt.plot(df2_info.index, df2_info['mean'], color= 'g', label= 'Mean HPI Growth', linewidth= 5, alpha= .7)

#1st standard deviation curves & shading
plt.plot(df2_info.index, df2_info['mean+std'], color= 'g', linestyle = '--', alpha= .45)
plt.plot(df2_info.index, df2_info['mean-std'], color= 'g', linestyle = '--', alpha= .45, label= '+/- 1 std')
plt.fill_between(x= df2_info.index, y1= df2_info['mean'], y2= df2_info['mean+std'], where= (df2_info['mean'] < df2_info['mean+std']), color= 'g', alpha= .3)
plt.fill_between(x= df2_info.index, y1= df2_info['mean'], y2= df2_info['mean-std'], where= (df2_info['mean'] > df2_info['mean-std']), color= 'g', alpha= .3)

#2nd standard deviation curves & shading
plt.plot(df2_info.index, df2_info['mean+2std'], color= 'g', linestyle = '-.', alpha= .35)
plt.plot(df2_info.index, df2_info['mean-2std'], color= 'g', linestyle = '-.', alpha= .35, label= '+/- 2 std')
plt.fill_between(x= df2_info.index, y1= df2_info['mean'], y2= df2_info['mean+2std'], where= (df2_info['mean'] < df2_info['mean+2std']), color= 'g', alpha= .2)
plt.fill_between(x= df2_info.index, y1= df2_info['mean'], y2= df2_info['mean-2std'], where= (df2_info['mean'] > df2_info['mean-2std']), color= 'g', alpha= .2)

#3rd standard deviation curves & shading
plt.plot(df2_info.index, df2_info['mean+3std'], color= 'g', linestyle = 'dotted', alpha= .15)
plt.plot(df2_info.index, df2_info['mean-3std'], color= 'g', linestyle = 'dotted', alpha= .15, label= '+/- 3 std')
plt.fill_between(x= df2_info.index, y1= df2_info['mean'], y2= df2_info['mean+3std'], where= (df2_info['mean'] < df2_info['mean+3std']), color= 'g', alpha= .1)
plt.fill_between(x= df2_info.index, y1= df2_info['mean'], y2= df2_info['mean-3std'], where= (df2_info['mean'] > df2_info['mean-3std']), color= 'g', alpha= .1)

#datapoints of max growth county
plt.plot(df2.index, df2['Davidson_County_pct'], 'bo', markersize= 1.2) #notice we use df2 here, not df2_info
plt.annotate("Max HPI Growth\n(Davidson Co.)",
            xy=(df2_info.index[43], 42), xycoords= 'data',
            xytext=(df2_info.index[31], 46), textcoords= 'data',
            arrowprops=dict(arrowstyle= "->", connectionstyle= "arc3, rad= -0.3", color= 'b'), 
            )

#datapoints of min growth county
plt.plot(df2.index, df2['Weakley_County_pct'], 'ro', markersize= 1.2) #notice we use df2 here, not df2_info
plt.annotate("Min HPI Growth\n(Weakley Co.)",
            xy=(df2_info.index[52], 4), xycoords= 'data',
            xytext=(df2_info.index[42], -8), textcoords= 'data',
            arrowprops=dict( arrowstyle= "->", connectionstyle= "arc3, rad= 0.3", color= 'r'), 
            )

plt.title("Tennessee Housing Price Index (HPI)\nPost-Recession Recovery", fontsize= 25)
plt.xlabel("Year", fontsize= 20)
plt.ylabel("Percent(%)",fontsize= 20)
plt.tick_params(axis= 'x', labelsize= 'x-large', labelrotation= 30)
plt.tick_params(axis= 'y', labelsize= 'x-large')
plt.legend(fontsize= 'xx-large', numpoints= 4)
```


![png](/images/HPI_Matplotlib_plot.png)

Although this is not definitively correct, I find it to be useful thinking of this as an aeriel view of a "3-Dimensional Density Plot", where time is the additional variable.  
This isn't really that far-fetched either. I say this because, in our case, the HPI data is normally distributed at each point along the x-axis. Implying that, by the Empirical Rule, the opacity of each shaded region accurately depicts of the density of data in that interval.  
  
Also plotted are the individual data points of the two counties at either end of the distribution. Davidson County (aka Nashville) nearly doubled its HPI, having the highest overall increase of 93%; the lowest in the state was Weakley County only increasing a minor 9%.  
  
Later on in this analysis, we will look deeper into some of the underlying aspects of Nashvilles sudden growth in popularity.  
## An Interactive Choropleth Map With Plotly 
Thus far, we've:
1. graphed the 1996-2018 HPI curves for all 88 counties 
2. used summary statistics to build an illustration of the post-recession HPI growth in Tennessee.  
  
But still yet, neither figure is successful in showing both individual county data *and* the Tennessee housing market as a whole. The first graph contains so much detail that practically all detail is lost, and our second figure lacks detail at the county level. This happens to be the most common issue with static figures.  
  
To include both individual county data and the housing market in Tennessee as a whole, we look towards the extremely powerful and interactive data analysis toolset found in the Plotly library. Specifically, we will be making an interactive thematic geo-map of the state.  
  
To do this, we need to make a few adjustments to our current DataFrame. First, and most importantly, we need the Federal Information Processing Standards (FIPS) county specific codes, which we will use later to define the geographical boundaries of each county on our map.


```python
import plotly.express as px
import plotly.figure_factory as ff

tn_locations = pd.read_excel("/Users/DrewWoods/Desktop/Py_Project_1/US_FIPS_GEOcodes.xlsx", sheet_name= 'TNcounty_FIPS')
tn_locations.set_index('County_PCT', inplace=True)

# join DataFrames to match county names to their FIPS codes 
df2 = df2.transpose()
df2.index = tn_locations.index
df2 = df2.join(tn_locations, on= df2.index)

# rearrange column order
cols = df2.columns.tolist()
cols = cols[-1:]+cols[-2:-1]+cols[:-2]
df2 = df2[cols]
```

Plotly figures function more smoothly with "tidy data" having few columns and many rows, so we tidy up our dataframe using the Pandas melt method.


```python
#tidying df2 for use in Plotly figure
cols_to_melt = df2.columns.values[2:]
df2 = pd.melt(df2, id_vars= ['FIPS_code', 'County_Name'],
             value_vars = cols_to_melt, 
             var_name = 'Date', 
             value_name= 'Percent_Change')

df2.sort_values(['County_Name', 'Date'],inplace = True)
df2.reset_index(drop = True, inplace= True)
```

We will format our first dataframe this way too so that we can include the original HPI data in the figure as well.


```python
# tidying df to insert 'HPI' column into df2
df = df.transpose()
df.reset_index(inplace= True)

df = pd.melt(df, id_vars= 'index',
                value_vars = df.columns.values[1:],
                var_name = 'Date',
                value_name= 'HPI')

df.sort_values(['index', 'Date'],inplace = True)
df.reset_index(drop=True, inplace=True)

# adding 'HPI' column
df2['HPI'] = df['HPI'].astype(int) 

# fixing column order again...
cols = df2.columns.to_list()
cols = cols[0:3] + cols[-1:] + cols[-2:-1]
df2 = df2[cols]
```

One downside to the choropleth map is that it can be very computationally expensive. The more traces your figure has, the more times the algorithm has to itterate through the data, matching the county names to the data, to the FIPS codes, and then the FIPS codes to the correct latitude and longitude coordinates that define the county boundaries. Since our map will be interactive, it will have a **lot** of traces.  
  
To mitigate the number of traces and the complexity of our figure, we reduce the number of observations we pass to it.


```python
biyearly_dates = []

for i in range(12,19):
    if i < 10:
        may = '200'+str(i)+'-05-31'
        sept = '200'+str(i)+'-09-30'
    else:
        may = '20'+str(i)+'-05-31'
        sept = '20'+str(i)+'-09-30'
    biyearly_dates.append(may)
    biyearly_dates.append(sept)

biyearly_dates = pd.Series(biyearly_dates).astype(dtype=str)
df2['Date'] = df2['Date'].astype(dtype=str)

dfFig = df2[df2['Date'].isin(biyearly_dates)]
dfFig.reset_index(drop = True, inplace = True)
```


```python
dfFig
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
      <th>FIPS_code</th>
      <th>County_Name</th>
      <th>Date</th>
      <th>HPI</th>
      <th>Percent_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>47001</td>
      <td>Anderson County</td>
      <td>2012-05-31</td>
      <td>106800</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>47001</td>
      <td>Anderson County</td>
      <td>2012-09-30</td>
      <td>106800</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>47001</td>
      <td>Anderson County</td>
      <td>2013-05-31</td>
      <td>109000</td>
      <td>2.1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>47001</td>
      <td>Anderson County</td>
      <td>2013-09-30</td>
      <td>108200</td>
      <td>1.3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>47001</td>
      <td>Anderson County</td>
      <td>2014-05-31</td>
      <td>110300</td>
      <td>3.3</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1227</td>
      <td>47189</td>
      <td>Wilson County</td>
      <td>2016-09-30</td>
      <td>237600</td>
      <td>39.1</td>
    </tr>
    <tr>
      <td>1228</td>
      <td>47189</td>
      <td>Wilson County</td>
      <td>2017-05-31</td>
      <td>256600</td>
      <td>50.2</td>
    </tr>
    <tr>
      <td>1229</td>
      <td>47189</td>
      <td>Wilson County</td>
      <td>2017-09-30</td>
      <td>260100</td>
      <td>52.3</td>
    </tr>
    <tr>
      <td>1230</td>
      <td>47189</td>
      <td>Wilson County</td>
      <td>2018-05-31</td>
      <td>272900</td>
      <td>59.8</td>
    </tr>
    <tr>
      <td>1231</td>
      <td>47189</td>
      <td>Wilson County</td>
      <td>2018-09-30</td>
      <td>278500</td>
      <td>63.1</td>
    </tr>
  </tbody>
</table>
<p>1232 rows × 5 columns</p>
</div>



Now that the figure data is prepared and is stored in a new DataFrame, all that remains is to bring in a json file with the coordinates for the counties!


```python
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
fig = px.choropleth(
        dfFig, 
        geojson= counties,                    
        scope='usa',                 
        center= dict(lat=35.860119,lon=-86.660156),
        locations= 'FIPS_code',
        color= 'Percent_Change',
        hover_name= 'County_Name',
        hover_data=['HPI'],
        animation_frame= 'Date',
        animation_group= 'FIPS_code',
        color_continuous_scale = 'thermal',
        range_color=(-10, 93)
                   )
fig.update_geos(fitbounds = "locations", visible = False)
fig.show()
```

{% include notebook path="/assets/html_files/TennHPI_Choropleth.html" %}


As you can see, making use of plots with interactive widgets is a great way to get the most out of your visualization. In our figure, we can use the date slider to choose which data we want to see, watch the map evolve as it cycles through the dates autonomously, and if you see any given county's data by simply hovering your mouse over it!  
  
  
We saw in our last graph how rapidly the HPI is increasing in Davidson County, but here we can see that not only the bordering counties, but every county in middle Tennessee is being influenced by Nashville's growth...  
If home values have nearly doubled in less than a decade, does that mean that income has as well? How long can this velocity be sustained?  
  
In the continuation of this analysis, we dig deeper into a few of the underlying reasons of  Nashville's rapid HPI growth and attempt to answer a few of these questions.  

