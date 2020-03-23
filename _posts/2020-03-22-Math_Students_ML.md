---
title: "Machine Learning for Predicting Student Success"
date: 2020-03-22
tags: [Data Analysis & Visualization] # Make Changes
breadcrumbs: true
header:
    overlay_image: "/images/nash-header.jpg"
    caption: "Photo credit: [**Crovean**](https://www.flickr.com/people/crovean/)"
    actions:
      - label: "More Info"
        url: "https://thda.org/research-planning/tennessee-housing-market"
excerpt: "Data Analysis & Visualizations"

toc: true
toc_label: " Workflow :"
toc_icon: "hiking"
---
I was always good at math growing up, but never really enjoyed it. It somehow became my passion early in my college career, and I decided to abandon pre-med to pursue a degree in actuarial science. I soon began taking notice of how everybody either **loved** math, or they absolutely dreaded it and would say something along the lines of:  
  
"I'm just not a math person"  
  
I've always assumed that these people just never truly gave themselves the oppurtunity to love mathematics because they never really tried to to do well and thoroughly conceptualize the material. But then I got thinking, is the statement "I'm just not good at math" a legitimate explanation to their failing grades? *Is mathematical ability genetic?*  
  
In this project, I am looking to build a model that takes seemingly irrelevant details about a student and uses them to predict their mathematical ability. 


# Data Exploration
This is a dataset from the UCI repository that contains the final scores of 395 students at the end of a math program with several features that may or may not impact the future outcome of these young adults. The dataset consists of 30 predictive features, and columns for two term grades the final grade (G1, G2, and G3, respectively). The 30 features detail the social lives, home lives, family dynamics, and the future aspirations of the students. The two term grades, G1 and G2, will not be included in the actual analysis since they essentially compose the final grade.  
  
  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Outline:**
1.  Variable Identification
2.  Univariate Analysis
3.  Bivariate Analysis
   * Target Correlation
   * Feature Correlation

## Variable Identification

The 30 predictive features are all categorical, and consist of both ordinal and nominal data.  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Ordinal Features:**
1.	age - student's age (numeric: from 15 to 22) 
2.	Medu - mother's education (numeric: 0 – none, 1 - primary education (4th grade), 2-  5th to 9th grade, 3-  secondary education, or 4- higher education) 
3.	Fedu - father's education (numeric: 0 – none, 1 - primary education (4th grade), 2-  5th to 9th grade, 3-  secondary education, or 4- higher education) 
4.	traveltime - home to school travel time (numeric: 1 - 1 hour) 
5.	studytime - weekly study time (numeric: 1 - 10 hours) 
6.	failures - number of past class failures (numeric: n if 1<=n<3, else 4) 
7.	famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent) 
8.	freetime - free time after school (numeric: from 1 - very low to 5 - very high) 
9.	goout - going out with friends (numeric: from 1 - very low to 5 - very high) 
10.	Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
11.	Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high) 
12.	health - current health status (numeric: from 1 - very bad to 5 - very good) 
13.	absences - number of school absences (numeric: from 0 to 93)  
  
  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Nominal Features:**
1.	school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2.	higher - wants to take higher education (binary: yes or no)
3.  sex - student's sex (binary: 'F' - female or 'M' - male)
3.	address - student's home address type (binary: 'U' - urban or 'R' - rural)
4.	famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
5.	Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
6.	schoolsup - extra educational support (binary: yes or no)
7.	famsup - family educational support (binary: yes or no)
8.  paid - extra paid classes within the course subject (binary: yes or no)
9.	activities - extra-curricular activities (binary: yes or no)
10.	nursery - attended nursery school (binary: yes or no)
11.	internet - Internet access at home (binary: yes or no)
12.	romantic - with a romantic relationship (binary: yes or no)
14. Mjob - mother's job (nominal: 'teacher', 'health', civil 'services', 'athome', or 'other') 
15. Fjob - father's job (nominal: 'teacher', 'health', civil 'services', 'athome', or 'other') 
16. reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other') 
17. guardian - student's guardian (nominal: 'mother', 'father' or 'other')  
  
  
  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Grade Columns:**
1.  G1 - first term grade (numeric: from 0 to 20) 
2.  G2 - second term grade (numeric: from 0 to 20) 
3.  G3 - final grade (numeric: from 0 to 20, output target)  
  
  
**At the top of the page is a link that will redirect you to the Kaggle post where this dataset can be found!**

## Univariate Analysis
The ordinal features are all integer values, and the nominal features are all strings. Let's take a look at the distributions of the 13 ordinal features.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load in dataset
stud = pd.read_csv('student-math.csv')
```


```python
plt.style.use('ggplot')
stud.hist(figsize= (20,15), color= 'b');
```

<p align="center">
  <img src="/images/math_ML_imgs/output_7_0.png">
</p>


Right away we can see that some of the features have categories with hardly any occurances (ex: Fedu, Medu, age, and absences), which isn't much of a surprise having a small sample size. These outliers may cause issues later on down the road since they will likely overfit the model; I will take note of this now, and will evaluate them a little deeoer in the upcoming sections.  
  
The shift in distributions of the three grades is rather interesting. You can see how the grades were more spread out and typically worse in the first term (G1), the students started to pick their grades up a bit in the second term (G2), and then the final grade (G3) resembles a normal distributaion (excluding the 0 values) with mean ~11.

## Bivariate Analysis

I'm building a classifier to predict whether a student will either pass or fail the course, so I need to define a threshold for G3 (which ranges from 0-20). The minimum passing grade in the U.S. education system is typically considered to be D minus. In the case at hand, any student whose G3 value is less than 12 will be classified as failed, and the rest are classified as passing.  
(**Note:** In this section the target labels are "pass"/"fail", but afterwards they will just be 0's and 1's.)


```python
stud['PASS/FAIL'] = stud['G3'].apply(lambda x: 'FAIL' if x<12 else 'PASS')
```

<h3><center>Target Correlation</center></h3>

First, I want to evaluate how the individual features correlate to the target variable. I will use seaborn to help visualize the pass/fail frequencies of each feature.  
  
You may have noticed that I only included the ordinal features in the previous section. To clarify, I did this because many of the nominal features only have two categories and they can be observed independently very easily in bivariate plots.  
  
**Note that** in my study, I evaluated all 30 features alongside the target variable. However, many of the features yeild little-to-no correlation with the target variable, and were simply not interesting -- the ones that *are interesting* are plotted below!  
  
**The Interesting Ordinal Features**

![](/images/math_ML_imgs/output_12_0.png) ![](/images/math_ML_imgs/output_12_1.png)

![](/images/math_ML_imgs/output_12_2.png) ![](/images/math_ML_imgs/output_12_3.png)
  
**The Interesting Nominal Features**

![](/images/math_ML_imgs/output_13_0.png) ![](/images/math_ML_imgs/output_13_1.png) ![](/images/math_ML_imgs/output_13_2.png)


It's not too surprising that there's not any individual features that directly correlate to the target variable. That is, a student's pass/fail status is the outcome of an entire semester's work, it's more complex than just a single test grade. Similarly, making a "pass" or "fail" prediction for a student is more complex than just looking at one single detail of their life!  
With that being said, I suspect the individual features will show more correlation with the actual 0-20 final grade, G3. For example, being in a romantic relationship probably won't cause you to fail all of your classes, but it may very easily affect your final grade by a few points.  

### Feature Correlation
Now we'll take a look at how all of the features correlate with eachother as well. I will first need to encode the nominal feature values with numerical representations.


```python
stud = stud.drop(columns= ['G1', 'G2', 'PASS/FAIL'])
stud['target'] = stud['G3'].apply(lambda x: 0 if x<12 else 1)
```


```python
# encoding boolean categorical features with binary values

# school--> [0=GP, 1=MS]
stud['school'] = stud['school'].apply(lambda x: 0 if x=='GP' else 1)
# sex--> [0=F, 1=M]
stud['sex'] = stud['sex'].apply(lambda x: 0 if x=='F' else 1)
# address--> [0=U, 1=R]
stud['address'] = stud['address'].apply(lambda x: 0 if x=='U' else 1)
# famsize--> [0=LE3, 1=GT3]
stud['famsize'] = stud['famsize'].apply(lambda x: 0 if x=='LE3' else 1)
# Pstatus--> [0=T, 1=A]
stud['Pstatus'] = stud['Pstatus'].apply(lambda x: 0 if x=='T' else 1)

# yes/no boolean features--> [0=no, 1=yes]
yesno_feats = ['schoolsup','famsup','paid','activities','nursery','internet','romantic','higher']
for feat in yesno_feats:
    stud[feat] = stud[feat].apply(lambda x: 0 if x=='no' else 1)
```


```python
# encoding multi-categorical features (alphabetically) to integer values
nominal_cols = ['Mjob','Fjob','reason', 'guardian']

for col in nominal_cols:
    stud[col] = stud[col].astype('category').cat.codes
```


```python
# feature correlation to target variable
print(stud.corr()['target'].sort_values(ascending= False))
```

    target        1.000000
    G3            0.730890
    Medu          0.181728
    Fedu          0.144512
    sex           0.116555
    higher        0.098660
    reason        0.092036
    internet      0.069929
    studytime     0.069135
    Mjob          0.044176
    Fjob          0.038649
    freetime      0.019907
    activities    0.016108
    nursery       0.015556
    famrel        0.011626
    Pstatus       0.003119
    paid         -0.002406
    romantic     -0.023315
    health       -0.043807
    famsup       -0.055478
    school       -0.062031
    Dalc         -0.063204
    famsize      -0.070946
    guardian     -0.078864
    address      -0.100083
    Walc         -0.104700
    traveltime   -0.107819
    absences     -0.111943
    goout        -0.127932
    age          -0.140488
    schoolsup    -0.182913
    failures     -0.257365
    Name: target, dtype: float64



```python
# feature correlation to final grade
print(stud.corr()['G3'].sort_values(ascending= False))
```

    G3            1.000000
    target        0.730890
    Medu          0.217147
    higher        0.182465
    Fedu          0.152457
    reason        0.121994
    sex           0.103456
    Mjob          0.102082
    paid          0.101996
    internet      0.098483
    studytime     0.097820
    Pstatus       0.058009
    nursery       0.051568
    famrel        0.051363
    Fjob          0.042286
    absences      0.034247
    activities    0.016100
    freetime      0.011307
    famsup       -0.039157
    school       -0.045017
    Walc         -0.051939
    Dalc         -0.054660
    health       -0.061335
    guardian     -0.070109
    famsize      -0.081407
    schoolsup    -0.082788
    address      -0.105756
    traveltime   -0.117142
    romantic     -0.129970
    goout        -0.132791
    age          -0.161579
    failures     -0.360415
    Name: G3, dtype: float64


To emphasize my previous statement, notice that the order of correlation with the actual final grade seems a bit more logical than with the 0-1 target variable.
  
  
To see all correlations, I will use seaborn to make a heatmap of the full correlation matrix!


```python
# sorted correlation matrix
G3_corr = stud.drop(columns= 'target').corr().sort_values(by= 'G3', ascending= False)
G3_corr = G3_corr[G3_corr.index.values]

# heatmap
plt.figure(figsize = (13,10))
sns.heatmap(G3_corr, 
        vmax= .3,
        vmin= -.3,
        center= 0,
        xticklabels= 1, 
        yticklabels= 1).set_title("Correlation Matrix");
```

<p align="center">
  <img src="/images/math_ML_imgs/output_22_0.png">
</p>



**Positive Correlations:**
1. 'address' and 'traveltime'
  
2. 'Dalc', 'Walc', 'goout', and 'freetime'
  
3. 'famsup' and 'paid'
  
  
**Negative Correlations:**
1. 'Dalc', 'Walc', and 'studytime'
  
2. 'studytime' and 'sex'  
  
These correlations are rather intuitive, but the inverse correlation between the sex of a student and their average time spent studying is worth noting. As a male, I will openly admit I'm not too surprised that we fall short to our female counterparts in this area. But what's intersting is that..


```python
print("On average, females spend",
      np.round((stud[stud['sex'] == 0]['studytime'].mean()/stud[stud['sex'] == 1]['studytime'].mean()-1)*100, 2),
      "% longer each week studying than males do. \nHowever,", 
      np.round((stud[stud['sex'] == 1]['target'].sum()/stud[stud['sex'] == 1]['sex'].count())*100, 2), 
     "% of males scored a passing grade, while the female percentage was only",
     np.round((stud[stud['sex'] == 0]['target'].sum()/stud[stud['sex'] == 0]['sex'].count())*100, 2), 
     "%.")
```

    On average, females spend 29.13 % longer each week studying than males do. 
    However, 47.06 % of males scored a passing grade, while the female percentage was only 35.58 %.


# Data Cleaning
With only 395 samples, I'd like to retain as many of them as possible. However, including the underrepresented categories we saw in the univariate analysis would ultimately result in a poor model.  
Let's take another look at them.


```python
# mother and father education level counts
print(stud['Medu'].value_counts())
print(stud['Fedu'].value_counts())
```

    4    131
    2    103
    3     99
    1     59
    0      3
    Name: Medu, dtype: int64
    2    115
    3    100
    4     96
    1     82
    0      2
    Name: Fedu, dtype: int64


I will have to exclude the samples with 0 values for Medu/Fedu entirely since they are vastly in the minority, even in comparison to the next smallest categories.  
  
  
**Note:** In the following graphs, the green bars denote the number of students who passed, and the red who failed.


```python
# student age counts
print(stud['age'].value_counts())
sns.catplot(x = 'age', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'Age');
```

    16    104
    17     98
    18     82
    15     82
    19     24
    20      3
    22      1
    21      1
    Name: age, dtype: int64


<p align="center">
  <img src="/images/math_ML_imgs/output_28_1.png">
</p>


I am inclined to keep the samples whose age is >=20 since there is a definite correlation between the students' age and pass rate. Yet it would be far-fetched to consider samples of sizes three, one, and one as accuracte representations of *any* populations.


```python
print(stud['absences'].value_counts())
sns.catplot(x = 'absences', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2', height= 7, aspect= 2).set(title = 'Absences')
plt.figure(figsize= (40,40))
```

    0     115
    2      65
    4      53
    6      31
    8      22
    10     17
    14     12
    12     12
    3       8
    7       7
    16      7
    18      5
    5       5
    20      4
    22      3
    13      3
    1       3
    9       3
    11      3
    15      3
    23      1
    24      1
    21      1
    25      1
    56      1
    26      1
    28      1
    30      1
    17      1
    38      1
    40      1
    54      1
    19      1
    75      1
    Name: absences, dtype: int64

<p align="center">
  <img src="/images/math_ML_imgs/output_30_2.png">
</p>



For absences, there are 46 samples belonging to categories with no more than 5 observations each, so it's not practical to simply just drop them. However, we can see that every instance of 25+ absences resulted in the same outcome (a failed course, shocker!), and hence we can bin these values together. Although this only accounts for nine of the underrepresented categories, it will still greatly improve the quality of the feature as a whole.

  
It only seems natural to use '25' to denote the 25+ bin, but using a slighly larger integer, say 30, may benefit certain models where the Euclidean distance of the datapoints are of importance.


```python
stud = stud[stud['Medu'] != 0]
stud = stud[stud['Fedu'] != 0]
stud = stud[stud['age'] != 20]
stud = stud[stud['age'] != 21]
stud = stud[stud['age'] != 22]
stud['absences'] = stud['absences'].replace({25:30,
                                            26:30,
                                            28:30,
                                            30:30,
                                            38:30,
                                            40:30,
                                            54:30,
                                            56:30,
                                            75:30})
```

A few other features stood out to me while evaluating the bivariate graphs that I'd like to see again. 


```python
sns.catplot(x = 'school', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'School')
sns.catplot(x = 'Pstatus', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'Parents: Together/Apart')
sns.catplot(x = 'higher', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'Higher Education');
```

<p align="center">
  <img src="/images/math_ML_imgs/output_34_0.png">
</p>



<p align="center">
  <img src="/images/math_ML_imgs/output_34_1.png">
</p>



<p align="center">
  <img src="/images/math_ML_imgs/output_34_2.png">
</p>



Each of these features have a dominate category by roughly 90% selection, so there's not much, if any, information to be gained by their inclusion. 

Consider the feature of whether or not a student plans on furthering their education, 'higher'. If a student does not plan on furthering their education, many ML models would predecit that they are going to fail the course solely due to the lack of information (18 students to 367).
  
These features will not be included in the model.


```python
stud = stud.drop(columns=['school', 'Pstatus', 'higher'])
```


```python
stud.to_csv("stud_FeatEngineer_1.csv")
```

There are still a few features I feel like would be irrelevant to passing a math class, but at this point, I cannot confirm that any one feature won't be useful to my model.  
  
Furthermore, the resulting dataset has 27 features, and 385 samples.  

# Model Selection
* K-Nearest Neighbors
* Suport Vector Classifier
* Logistic Regression
* Decision Tree Classifier
* Gaussian Naive Bayes 
* Random Forest Classifier
* Gradient Boosting Classifier   
  
I will revisit feature selection soon, but for now I will evaluate the base model performances of the seven classifiers above, and will move forward with the top two.  
  
**If you've made it this far, you deserve a little honesty from me..**  
Prior to this very moment, the only exposure I've had in machine learning was roughly 2 class periods of mathematical statistics junior year when we slightly touched on linear regression. I'm sure there has to be better ways to go about selecting the best model for my problem, but I'm eager to learn, and I'm taking this project entirely as a learning experience. Plus, this way I'll have two different models that I know will at least perform semi-decent that I can study head to toe, research, and build. :-)


```python
stud = pd.read_csv("stud_FeatEngineer_1.csv", index_col= False)
stud = stud.drop(columns= 'Unnamed: 0')
stud
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
      <th>sex</th>
      <th>address</th>
      <th>famsize</th>
      <th>schoolsup</th>
      <th>famsup</th>
      <th>paid</th>
      <th>activities</th>
      <th>nursery</th>
      <th>internet</th>
      <th>romantic</th>
      <th>age</th>
      <th>Medu</th>
      <th>Fedu</th>
      <th>traveltime</th>
      <th>studytime</th>
      <th>failures</th>
      <th>famrel</th>
      <th>freetime</th>
      <th>goout</th>
      <th>Dalc</th>
      <th>Walc</th>
      <th>health</th>
      <th>absences</th>
      <th>Mjob</th>
      <th>Fjob</th>
      <th>reason</th>
      <th>guardian</th>
      <th>G3</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>U</td>
      <td>GT3</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>18</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>at_home</td>
      <td>teacher</td>
      <td>course</td>
      <td>mother</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>U</td>
      <td>GT3</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>at_home</td>
      <td>other</td>
      <td>course</td>
      <td>father</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>U</td>
      <td>LE3</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>at_home</td>
      <td>other</td>
      <td>other</td>
      <td>mother</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>U</td>
      <td>GT3</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>yes</td>
      <td>15</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>health</td>
      <td>services</td>
      <td>home</td>
      <td>mother</td>
      <td>15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>U</td>
      <td>GT3</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>16</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>other</td>
      <td>other</td>
      <td>home</td>
      <td>father</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>380</th>
      <td>F</td>
      <td>U</td>
      <td>LE3</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>18</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>teacher</td>
      <td>services</td>
      <td>course</td>
      <td>mother</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>F</td>
      <td>U</td>
      <td>GT3</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>no</td>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>other</td>
      <td>other</td>
      <td>course</td>
      <td>mother</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>382</th>
      <td>M</td>
      <td>U</td>
      <td>LE3</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>17</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>services</td>
      <td>services</td>
      <td>course</td>
      <td>mother</td>
      <td>16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>383</th>
      <td>M</td>
      <td>R</td>
      <td>LE3</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>18</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>services</td>
      <td>other</td>
      <td>course</td>
      <td>mother</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>384</th>
      <td>M</td>
      <td>U</td>
      <td>LE3</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>no</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>other</td>
      <td>at_home</td>
      <td>course</td>
      <td>father</td>
      <td>9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>385 rows × 29 columns</p>
</div>




```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# instantiating the models within a list
models = []
models.append(('KNN', KNeighborsClassifier())) 
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
```

Although this dataset is from Kaggle, I'm not looking for a model that optimizes for accuracy at the expense of generality. I'm genuinely interested in the predictive capabilities of the 27 features that remain, and would like my model to be able to predict on new data as well.  
For this to happen, I must estabalish a few precautionary measures.  
  
**High Variance**  
The dataset contains a less than preferrable amount of samples, and because of this, the testing accuracy will be highly subject to variation. I want the two best performing models of the group, but I do not want the best two if they can't fit to new data. In attempt to mitigate this, I will use 10 repititions of 5-fold stratified cross validation to estimate each model's out-of-sample accuracy.  
  
**Data Leakage**  
Instead of passing the actual models to cross_val_score, I will cross validate pipelines which contain the models **and** the preprocessing steps. Thus, the dataset will be split into new training/test sets at each fold, then will be preprocessed separately, and the chances of data leakage will be near-to-none!


```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

# seperating target variable from feature set
Y = stud['target']
X = stud.drop(columns=['G3', 'target'])

# list of column labels with string values
# list of column labels with int values
cat_feats = X.select_dtypes('object').columns.to_list()
num_feats = X.select_dtypes('int').columns.to_list()

# 2-stage preprocessor
preprocessor = make_column_transformer(
    (MinMaxScaler(), num_feats),
    (OneHotEncoder(drop= 'first'), cat_feats))

# computing and storing each model's average CV accuracy
names = []
scores = []
for name, model in models:
    pipe = make_pipeline(preprocessor, model)
    rep_kfold = RepeatedStratifiedKFold(n_splits= 5, n_repeats= 10, random_state= 777)
    score = cross_val_score(pipe, X, Y, cv=rep_kfold, scoring='accuracy').mean() 
    
    names.append(name)
    scores.append(score)
    
# bar chart displaying the results
model_CVscores = pd.DataFrame({'Name': names, 'Score': scores})
model_CVscores.sort_values(by= 'Score', ascending= False).plot(kind='bar', 
                                                               x= 'Name', 
                                                               y= 'Score', 
                                                               style= 'ggplot', 
                                                               rot= 0, 
                                                               legend= False)

print(model_CVscores.sort_values(by= 'Score', ascending= False))
```

      Name     Score
    2   LR  0.658182
    5   RF  0.651948
    1  SVC  0.636364
    4  GNB  0.628312
    6   GB  0.622078
    3   DT  0.591429
    0  KNN  0.566234


<p align="center">
  <img src="/images/math_ML_imgs/output_43_1.png">
</p>


Logistic Regession and Random Forest seem to be the best two classifiers both scoring over 65% accuracy right out of the box with default parameters, minimal data cleaning, and no feature selection.