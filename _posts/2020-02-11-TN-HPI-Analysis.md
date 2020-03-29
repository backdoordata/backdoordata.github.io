---
title: "Tennessee Housing Market: Post-Recession"
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
toc_sticky: true
---

# <center>HPI Data</center>
If you live in Tennessee, you've probably noticed the growth in popularity of Middle Tennessee. If you don't live in Tennessee, or you've been completely oblivious to this fact, you've come to the right place!  
  
My aim for this project is to familiarize myself with the Pandas library, a few of the popular visualization libraries in Python, and to show the change in Tennessee since the '08 recession using historical Housing Price Index (HPI) data. The data I used in this project was collected by Zillow, which I queried from the Quandl database.  

## Query
On the Quandl website, you'll find lots of open-source data that can be easily accessed with a free account. To grab the data for all 95 Tennessee counties, I made a query for each individual county dataset by itteratively calling the quandl.get method on the respective URL's, and then merging them to a pandas dataframe at the end of each iteration. Since all of the data is of the same measure and same state, the URL's only differed by a 3-4 digit string, which can easily be scraped from the [documentation page](https://www.quandl.com/data/ZILLOW-Zillow-Real-Estate-Research/documentation).  
  
**Note that if you wish to use this method, the native column labels will be URL's, so be sure to have the county names at hand in the same order that you queried the data!*  
  
## Data Preparation
Zillow does a good job structuring their data and making it easily explorable, so there wasn't much cleaning needed for our datset to become workable except for assessing the NaN values! Seven county datasets weren't successfully queried, so I suppose Zillow doesn't service those counties. I'll continue with the remaining 88 counties.
  
The missing entries occur at either the beginning dates or the end dates. The dataset contains HPI data from 1996-2018, but since I'm only using the data *after* the 2008 recession, the missing entries from before then aren't an issue. However, I went ahead and filled them as well incase I want to revisit the dataset.  
As you'll soon see, the county HPI's are highly correlated. Taking advantage of this, I can use the data from a similar county to fill the missing entries of another county.


```python
def Find_Most_Similar(county_name):
    """Sorts counties by similarity to county_name, prints top 5, and 
    returns a plot of county_name vs. the two most similar"""
    
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

The graph above is not exactly ideal; it's so detailed that it's hard to see anything besides the obvious correlation between the counties. However, notice how the curves begin to uncluster after the '08 recession. This is what we want to focus on.

# <center>Making the Visualizations</center>
The Great Recession 'officially' ended in June of 2009, but as you can tell from the graph, the housing market continued to suffer. I want to know when the housing markets began recovering, and then use that as the new starting date.


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

The starting HPI value for each county will be used as a basis to measure their growth.

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
## Plotting Summary Statistics
To make a figure that's more informative than the previous one, I'll utilize the summary statistics. I created the figure below using Matplotlib's graphical plotting library!


<p align="center">
  <img src="/images/HPI_imgs/HPI_Matplotlib_plot.png">
</p>

Although this is not definitively correct, I find that it's useful to view this as an overhead 3-dimensional density plot, with time being the third variable.  
This isn't too far-fetched either since the data is normally distributed at each point on the x-axis. Using the Empirical Rule, each shaded regions degree of opacity accurately depicts the density.  
  
Also included is the individual monthly data for the counties at opposing ends of the distribution. Davidson County, or Nashville, nearly doubled in HPI (93% growth), while Weakley County's HPI only increased by 9%.  

## Building An Interactive Choropleth Map
To show both county specific data, **and** the entire Tennessee housing market, I'll utilize the powerful data analysis toolset found in the Plotly Express library. Specifically, I'll build an interactive thematic geo-map of Tennessee and its' counties.  
  
To define the geographical boundaries of the map, I used county specific codes from the Federal Information Processing Standards, called FIPS codes, and a json file that contains the correct latitude and longitude coordinates(which can be found [here](https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json)).  
Plotly likes "tidy" data with as few columns as possible. Having 88 columns, this took quite a bit of restructuring, which I did for both dataframes, old and new, in order to have the percent change data **and** the HPI data in the figure.  
  
The major downside to a choropleth map is that it can be very computationally demanding. Since this is an interactive map, it has to reiterate through the data constantly, introducing more latency with each update.  
  
To mitigate the complexity of the figure, I'll simply reduce the amount of data that's used in it.


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

The dataframe used in the figure below (dfFig) contains two observations for each year in 2012-2018 for each county, and has five columns labeled 'FIPS_code', 'County_Name', 'Date', 'HPI', and 'Percent_Change'.

  
{% include TennHPI_Choropleth.html %}
 
As you can see, you can manually select a date to view by using the slider, or you can click the play button to see the entire progression. You can also hover over any county to see its' data for the current date.

# Conclusion
Both visualizations, static and interactive, have their place in data analysis. In this project, the first graph excels by communicating a summary of the dataset very efficiently. It's highly interpretable, but it lacks the valuable details of each county. The interactive Plotly map summerizes the whole dataset, contains the individual county data, and is also easy to interpret. However, it's not something you'd present at a meeting or to a crowd of people since it relies on the viewer's input.  
  
To me, the most enjoyable aspect of making data visualizations is that there's never just one right answer, or one "go-to graph". The best visual to use entirely depends on how well you answer the questions:  
  
*  "Who's going to be viewing this visual?"  
*  "What setting will the intended audience be in?"  
*  "Do I need to sacrifice detail for simplicity?"  
*  "What is the most vital information to portray?"  
*  "How can I relay this information so that the results are undeniable, but without sacrificing integrity?"  




