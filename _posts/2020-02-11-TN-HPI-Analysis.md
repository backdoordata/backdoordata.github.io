---
title: "Visualizing Post-Recession Housing Markets"
date: 2020-02-11
tags: [Data Analysis & Visualization]
breadcrumbs: true
header:
    overlay_image: "/images/nash-header.jpg"
    caption: "Photo credit: [**Crovean**](https://www.flickr.com/people/crovean/)"
    actions:
      - label: "More Info"
        url: "https://thda.org/research-planning/tennessee-housing-market"
excerpt: "Data Analysis & Visualizations"

toc: true
toc_label: " Sights-to-See:"
toc_icon: "hiking"
---

If you live in Tennessee, you've probably noticed the recent growth in popularity of Middle Tennessee. If you don't live in Tennessee, or have been completely oblivious to this fact, you've come to the right place!
  
# The Data
The Housing Price Index (HPI) data used in this project was collected by Zillow, and can be found in the Quandl databases.  

## Query
On the Quandl website, you'll find lots of open-source data that can be easily accessed with a free account. To grab the data for all 95 Tennessee counties, I made a query for each individual county dataset by itteratively calling the quandl.get method on the respective URL's, and then merging them to a pandas dataframe at the end of each iteration. Since all of the data is of the same measure and same state, the URL's only differed by a 3-4 digit string, which can easily be scraped from the [documentation page](https://www.quandl.com/data/ZILLOW-Zillow-Real-Estate-Research/documentation).  
  
**Note that if you wish to use this method, the native column labels will be URL's, so be sure to have the county names at hand in the same order that you queried the data!*  
  
## Data Preparation
Zillow does a good job structuring their data and making it easily explorable, so there wasn't much cleaning needed for our datset to become workable except for assessing the NaN valued entries! Seven of the columns were completely missing since Zillow doesn't service those counties apparently, so I just dropped them from the dataframe.  
  
Next, I noticed some missing entries from either the beginning or end of the collection. Since our dataframe contains the monthly HPI from 1996-2018, and we are only using the latter portion for our visualization, the missing early entries weren't of much concern. However, I went ahead and filled them too incase I decide to revisit this dataset later.  
As you'll soon see, the counties are highly correlated with one another. Taking advantage of this, the following code approximates and fills the missing values.


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

  
Making a very basic plot, we can now take a peak at the dataset!  

<p align="center">
  <img src="/images/HPI_imgs/HPI_linegraph.png">
</p>

The graph above is not exactly ideal; it's so detailed that it's hard to see anything besides the obvious correlation between the counties. However, notice the increasing variation after the '08 recession as the HPI curves begin to uncluster. This is what we will focus on.

# Creating Visualizations
The Great Recession 'officially' ended in June of 2009, but as you can tell from the graph, the housing market continued to suffer. In the following lines of code, the dataframe is altered to only include years of post-recession recovery.


```python
df_09_plus = df.loc['2009-6-30':]
min_months = []

# creates list of post recession turning-point dates
for i in np.arange(0,88):
    itter_county = df_09_plus.iloc[:,i]
    local_min = min(itter_county)
    min_mo = df_09_plus.index.where(itter_county == local_min).dropna().values
    min_mo = pd.to_datetime(min_mo.item(0))
    min_months.append(min_mo)
    
# returns the most observed date from list 
# truncates df at that point
start_date = max(set(min_months), key=min_months.count)
df = df[df.index >= start_date]
```

As I'm not concerned with previous years, I'm also not concerned with the HPI values from those years. Rather, I will use percentages as a measure of growth and improvement. The following lines produce the desired dataframe.


```python
# create new dataframe with entries as %change in HPI from May 2012
pct_func = lambda x : round(100*(x/(df.iloc[0].values) - 1),1)
df2 = pct_func(df.iloc[:])

# relabel columns of the new dataframe
new_cols = []
for i in df2.columns:
    pct_col = i.strip('_HPI')+'_pct'
    new_cols = np.append(new_cols, pct_col)
    
df2.columns = new_cols
```
## Visualizing Post-Recession Growth
To make a figure more interpretable than a graph with 88 HPI curves, we can plot summary statistics instead! I created the figure below using the graphical plotting library in Matplotlib.


<p align="center">
  <img src="/images/HPI_imgs/HPI_Matplotlib_plot.png">
</p>

Although this is not definitively correct, I find it useful to think of this as an overhead view of a 3-Dimensional Density Plot, where time is the additional variable.  
This isn't too far-fetched either. I say this because the HPI data is normally distributed at each point along the x-axis. Implying that, by the Empirical Rule, the opacity of the shaded regions accurately depict the density of datapoints in each interval.  
  
Also included are the individual data points of the two counties at either end of the distribution. Davidson County (Nashville) had the highest overall increase of 93%, nearly doubling its HPI; the lowest in the state was Weakley County, only increasing 9%.  

## Building An Interactive Choropleth Map
To show individual county data **and** the housing market as a whole in a single figure, I utilized the extremely powerful data analysis toolset found in the Plotly Express library. Specifically, an interactive thematic geo-map of Tennessee partioned by its counties.  
  
To define the geographical boundaries of the map, I used county specific codes from the Federal Information Processing Standards (FIPS) and a json file containing the corresponding latitude and longitude strings, which can be found [here](https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json).  
Plotly likes "tidy" data with as few columns as possible. Having 88 columns, this took quite a bit of restructuring. I did this for both dataframes to include both the percent change and the HPI of each county in the figure.  
  
The major downside to a choropleth map is that it can be very computationally expensive. Since this is an interactive map, it has to reiterate through the data each frame, increasingly introducing more latency.  
  
To mitigate complexity of the figure, I reduced the number of observation dates we pass to it.


```python
biyearly_dates = []

for i in range(12,19):
    may = '20'+str(i)+'-05-31'
    sept = '20'+str(i)+'-09-30'

    biyearly_dates.append(may)
    biyearly_dates.append(sept)

biyearly_dates = pd.Series(biyearly_dates).astype(dtype=str)
df2['Date'] = df2['Date'].astype(dtype=str)

dfFig = df2[df2['Date'].isin(biyearly_dates)]
dfFig.reset_index(drop = True, inplace = True)
```

The dataframe used in the figure below, dfFig, consists of two observations for each year in 2012-2018, and the five columns: 'FIPS_code', 'County_Name', 'Date', 'HPI', 'Percent_Change'.

  
{% include TennHPI_Choropleth.html %}
 
  
# Conclusion
Both types of visualizations, static and interactive, have their place in data analysis. In this project, the static Matplotlib visualizations excels in relaying the bulk summary of the data very efficiently and quickly, but lacks valuable details of the individual counties. The interactive Plotly figure contains the individual datapoints and is also easily interpretable, but it wouldn't be the type of content you'd want to bring into a meeting or to present to a crowd since it entirely relies on user input. Aside from the Static vs. Interactive comparison, geographical data is almost ALWAYS best visualized using some form of a thematic map!  
  
Personally, I think the most enjoyable part of making visualizations is that there's never just one right answer, or one "Go-To" figure for all the different datasets. Sure it's easy to make some pretty atrocious visualizations, but there's a thousand different ways to skin a cat, right? It's up to you to determine which way's the best way.

