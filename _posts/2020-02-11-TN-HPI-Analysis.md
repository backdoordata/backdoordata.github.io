---
title: "Visualizing The Growth of Tennessee"
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

If you live in Tennessee also, you'll agree that it'd be impossible to ignore the recent growth of middle Tennessee. If you don't live in Tennessee (or have just been completely oblivious to this fact) you've come to the right place; hopefully this post will show you what I mean!

If you're only interested in a specific section, I included some shortcuts for you to the right of this text!
  
  
# The Data
The Housing Price Index (HPI) data used in this project was collected by Zillow, and can be found in the Quandl databases.  
  
All of the data offered by Quandl is open-source, and can be easily obtained with a free account. To grab the data for all 95 Tennessee counties, the method I used was to query each individual county dataset by itteratively calling the quandl.get method on the respective URL's, and then joining them into a pandas dataframe at the end of each iteration. Since all of the data is of the same measure, and just for one states counties, the URL's only differed by a 3-4 digit section, which can easily be scraped from the documentation page. More information on this can be found [here](https://www.quandl.com/data/ZILLOW-Zillow-Real-Estate-Research/documentation). Note that if you wish to use this method, the column labels will just be URL's, so make sure to have the county names at hand and in the same order that you queried the data!  
  
## Assessing NaN Values
From what I've seen, Zillow does a good job structuring their data and making it easily explorable. There wasn't much cleaning necessary for our datset to become workable except for assessing the NaN entries! A few of my columns were completely empty, and it turns out Zillow doesn't service those seven counties so I just dropped them from the dataframe.  
  
Next, I noticed some counties were missing entries from either the beginning of the collection, or from the end. Since our dataframe contains the monthly HPI from 1996-2018 and we are only using the latter portion our visualization, the early missing entries weren't of much concern. However, I went ahead and took care of them anyways incase I decide to revisit this dataset later on. As you'll soon see, the counties are highly correlated with one another when it comes to the housing market; taking advantage of this, I used the following code to approximate and fill the missing values:




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


Using Matplotlib's basic .plot() function, we can now take a peak at what the dataset looks like!  

![png](/images/HPI_linegraph.png)

# Creating Visualizations
The graph above is not what we want, it's so detailed that it's hard to see anything besides the correlation between the counties. However, you can already see the increasing variation in the later years as the HPI curves begin to uncluster after the '08 recession. To focus on this, we will look only at the recovering housing market.  
  
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

As we aren't concerned with the earlier years, we also aren't concerned with the existing values of each counties HPI. Rather, we use percentages as a basis to evaluate change.  
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
```
## Visualizing Post-Recession Growth
To avoid another graph of 88 HPI curves, we can plot summary statistics instead! The figure below was created using the graphical plotting library in the Matplotlib package.


![png](/images/HPI_Matplotlib_plot.png)

Although this is not definitively correct, I find it to be useful thinking of this as an aeriel view of a "3-Dimensional Density Plot", where time is the additional variable.  
This isn't too far-fetched either. I say this because, in our case, the HPI data is normally distributed at each point along the x-axis. Implying that, by the Empirical Rule, the opacity of each shaded region accurately depicts the density of points in that interval.  
  
Also included are the individual data points of the two counties at either end of the distribution. Davidson County (Nashville) had the highest overall increase of 93%, nearly doubling its HPI; the lowest in the state was Weakley County, only increasing 9%.  

## Building An Interactive Choropleth Map
To show individual county data **and** the housing market as a whole in a single figure, I utilized the extremely powerful and interactive data analysis toolset found in the Plotly Express library. Specifically, we will be making an interactive choropleth map, which is a thematic geo-map of Tennessee partioned by its counties.  
  
To define the geographical boundaries for the map, I used county specific codes from the Federal Information Processing Standards (FIPS) and a json file containing the corresponding latitude and longitude strings, which can be found [here](https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json).  
Plotly likes "tidy" data with as few columns as possible. Having 88 columns, this took quite a bit of restructuring. However, I did this for both dataframes so I could include both the HPI and the percent change of each county in the figure.  
  
The major downside to the choropleth map is that it can be very computationally expensive. Since this is an interactive map, it has to reitterate through the data each frame, increasingly introducing latency.  
  
To mitigate the complexity of our figure, I reduced the number of observation dates we pass to it.


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

The dataframe used in the figure below, dfFig, consists of two observations for each year in 2012-2018, and the five columns labeled: 'FIPS_code', 'County_Name', 'Date', 'HPI', 'Percent_Change'.


  
{% include TennHPI_Choropleth.html %}
 

