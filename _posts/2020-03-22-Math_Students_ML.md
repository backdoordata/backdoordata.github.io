---
title: "Predict Student Success with Machine Learning"
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
I was always good at math growing up, but never really enjoyed it. It somehow became my passion early in my college career, and I decided to abandon pre-med to pursue a degree in actuarial science. I soon began taking notice of how people either loved math, or they absolutely hated it and would say something along the lines of "I'm just not a math person".  
  
I've always assumed that these people just never gave themselves the oppurtunity to truly enjoy mathematics because they never legitimately tried to to do well, and thoroughly conceptualize the material. But then I got thinking, is "I'm just not a math person" a legitimate explanation for their failing grades? **Is mathematical ability genetic?** 
  
For this project, I am looking to build a model that takes seemingly irrelevant details about students, and uses them to predict their mathematical ability. 


<h1><center>Data Exploration</center></h1>
This is a dataset from the UCI repository that contains the final scores of 395 students at the end of a math program with several features that may or may not impact the future outcome of these young adults. The dataset consists of 30 predictive features, and columns for the two term grades and final grade (G1, G2, and G3, respectively). The 30 features detail the social lives, home lives, family dynamics, and the future aspirations of the students. For obvious reasons, the two term grades, G1 and G2, will not be included in the actual model.  
  
  
  
&nbsp;&nbsp;&nbsp;&nbsp; **Outline:**  
1.) Variable Identification  
2.) Univariate Analysis  
3.) Bivariate Analysis  
3a.) Target Correlation  
3b.) Feature Correlation  
  

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load in dataset
stud = pd.read_csv('student-math.csv')
```

<h2><center>Variable Identification</center></h2>

The 30 predictive features are all categorical, and contain a mix of numeric and nonnumeric data types.  
  
&nbsp;&nbsp;&nbsp;&nbsp; **Ordinal Features:**
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
  
  
  
&nbsp;&nbsp;&nbsp;&nbsp; **Nominal Features:**
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
  
  
  
  
&nbsp;&nbsp;&nbsp;&nbsp; **Grade Columns:**
1.  G1 - first term grade (numeric: from 0 to 20) 
2.  G2 - second term grade (numeric: from 0 to 20) 
3.  G3 - final grade (numeric: from 0 to 20, output target)  
  
  
**At the top of the page is a link that will redirect you to the Kaggle post where this dataset can be found!**

<h2><center>Univariate Analysis</center></h2>
The ordinal features are all integer values, and the nominal features are all strings. Let's take a look at the distributions of the numeric columns.


```python
plt.style.use('ggplot')
stud.hist(figsize= (20,15), color= 'b');
```

<p align="center">
  <img src="/images/math_ML_imgs/output_7_0.png">
</p>


We can see right away that some of the features have categories with hardly any occurances (ex: Fedu, Medu, age, and absences), which isn't a surprise having a small sample size. These outliers may cause issues later on down the road since they will likely overfit the model; I will take note of this now, and will evaluate them further in the upcoming sections.  
  
The shift in distributions of the three grades is rather interesting. You can see how the grades were more spread out and typically worse in the first term (G1), the students started to pick their grades up a bit in the second term (G2), and then the final grade (G3) resembles a normal distribution (excluding the 0 values) with mean ~11.

<h2><center>Bivariate Analysis</center></h2>

To predict whether a student will either pass or fail, I need to define a threshold for G3. The minimum passing grade in the U.S. is typically considered to be a D-, so in our case, a 12/20 is the threshold for a minimum passing grade. 
*(In this section the target labels are pass or fail, but in following sections they will just be 0's and 1's.)*


```python
stud['PASS/FAIL'] = stud['G3'].apply(lambda x: 'FAIL' if x<12 else 'PASS')
```

### Target Correlation

First, I want to evaluate how the individual features correlate with the target variable. I will use seaborn to visualize the pass/fail frequencies of each feature.  
  
You may have noticed that I only included the ordinal features in the previous section. To clarify, I did this because many of the nominal features only have two categories and they can be observed independently very easily in bivariate plots.  
  
**Note that** I did evaluate the entire feature set. However, many of the features have little-to-no correlation with the target variable and were uninteresting, so I did not include them here -- the ones that *are interesting* are plotted below!  
  
**Ordinal features...**

![](/images/math_ML_imgs/output_12_0.png) ![](/images/math_ML_imgs/output_12_1.png)

![](/images/math_ML_imgs/output_12_2.png) ![](/images/math_ML_imgs/output_12_3.png)
  
**Nominal features...**

![](/images/math_ML_imgs/output_13_0.png) ![](/images/math_ML_imgs/output_13_1.png) ![](/images/math_ML_imgs/output_13_2.png)


What's important to acknowldge here is that a student's pass/fail status is the outcome of an entire semester's work; it's more complex than just a single test grade. Similarly, making a pass/fail prediction for a student is more difficult than just evaluating one feature!  
With this being said, I suspect the features will show more correlation with the actual 0-20 final grade. *Being in a relationship with someone probably won't cause them to fail all of their classes, but it may very easily affect their final grade by a few points.*  

### Feature Correlation
Now we'll take a look at all corrolations. I'll first need to encode the features who have nonnumeric entries.


```python
# drop term grades and old pass/fail target
stud = stud.drop(columns= ['G1', 'G2', 'PASS/FAIL'])

# define new target column
stud['target'] = stud['G3'].apply(lambda x: 0 if x<12 else 1)

# encode boolean categorical's with binary values
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

# encoding multi-categorical features (alphabetically) to integer values
nominal_cols = ['Mjob','Fjob','reason', 'guardian']
for col in nominal_cols:
    stud[col] = stud[col].astype('category').cat.codes
```
To emphasize on my prior statement, notice how the order of correlation with the actual final grade seems more logical than with the 0-1 target variable.

```python
# feature correlation to target variable
print(stud.corr()['target'].sort_values(ascending= False))

# feature correlation to target variable
print(stud.corr()['G3'].sort_values(ascending= False))
```

![](/images/math_ML_imgs/target_corr.png) ![](/images/math_ML_imgs/G3_corr.png)


Thus, I'll use G3 for making a correlation matrix heatmap.

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
* 'address' and 'traveltime'
  
* 'Dalc', 'Walc', 'goout', and 'freetime'
  
* 'famsup' and 'paid'
  
  
**Negative Correlations:**
* 'Dalc', 'Walc', and 'studytime'
  
* 'studytime' and 'sex'  
  
These correlations are seem reasonable, but the inverse correlation between the sex of a student and their average time spent studying is interesting. I'll openly admit that I'm not too surprised that we fall short to our female counterparts in this area, but what's intersting is that..


```python
print("On average, females spend",
      np.round((stud[stud['sex'] == 0]['studytime'].mean()/stud[stud['sex'] == 1]['studytime'].mean()-1)*100, 2),
      "% longer each week studying than males do. \nHowever,", 
      np.round((stud[stud['sex'] == 1]['target'].sum()/stud[stud['sex'] == 1]['sex'].count())*100, 2), 
     "% of males scored a passing grade, while the female percentage was only",
     np.round((stud[stud['sex'] == 0]['target'].sum()/stud[stud['sex'] == 0]['sex'].count())*100, 2), 
     "%.")
```
![](/images/math_ML_imgs/sex_comparison.png)


<h1><center>Data Cleaning</center></h1>
With only 395 samples, I'd like to retain as many of them as possible. However, including underrepresented categories would ultimately result in a weak model.  
Let's take another look at the ones we saw earlier.


```python
# mother and father education level counts
print(stud['Medu'].value_counts())
print(stud['Fedu'].value_counts())
```

![](/images/math_ML_imgs/Medu_counts.png) ![](/images/math_ML_imgs/Fedu_counts.png)


I will have to exclude the samples with 0 values for Medu/Fedu entirely since they are greatly underrepresented, even in comparison to the next smallest categories.  


```python
# student age counts
print(stud['age'].value_counts())
sns.catplot(x = 'age', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'Age')
```

![](/images/math_ML_imgs/age_counts.png)

![](/images/math_ML_imgs/output_28_1.png)

It would be far-fetched to consider samples of sizes three, one, and one as accurate representations of *any* populations.  
This feature is an interesting one nonetheless.


```python
print(stud['absences'].value_counts())
sns.catplot(x = 'absences', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2', height= 7, aspect= 2).set(title = 'Absences')
plt.figure(figsize= (40,40))
```

![](/images/math_ML_imgs/absences_counts.png)

<p align="center">
  <img src="/images/math_ML_imgs/output_30_2.png">
</p>

For 'absences', there's 46 samples in categories with no more than 5 total observations, and it's not ideal to simply drop them all. However, we can see that every instance where 'absences' > 24 resulted in the same outcome, a failed course. Hence, we can bin these values together. Although this only accounts for a portion of the underrepresented categories, it will still greatly improve the quality of the feature as a whole.

  
It only seems natural to use '25' to denote the 25+ bin, but using a slighly larger integer, say 30, may benefit certain models where Euclidean distance is of importance.


```python
# drop outlying samples
stud = stud[stud['Medu'] != 0]
stud = stud[stud['Fedu'] != 0]
stud = stud[stud['age'] != 20]
stud = stud[stud['age'] != 21]
stud = stud[stud['age'] != 22]

# make outlying 'absences' bin
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

A few other features stood out to me while evaluating the bivariate graphs. 


```python
sns.catplot(x = 'school', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'School')
sns.catplot(x = 'Pstatus', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'Parents: Together/Apart')
sns.catplot(x = 'higher', data= stud, hue= 'target', kind= 'count', hue_order= [1, 0], palette= 'Set2').set(title = 'Higher Education');
```

![](/images/math_ML_imgs/output_34_0.png) ![](/images/math_ML_imgs/output_34_1.png) ![](/images/math_ML_imgs/output_34_2.png)


Each of these features have a dominate category with roughly 90% selection, so there's not much, if any, information to be gained by their inclusion. 

That is, consider the feature 'higher'. If a student does not plan on furthering their education, many ML models would predecit that they would fail the course solely due to lack of information (18 students to 367).  
These features will not be included in the model.


```python
stud = stud.drop(columns=['school', 'Pstatus', 'higher'])
```

The resulting dataset has 27 features and 385 samples.  
  
There are still a few features I feel aren't relevant to passing a math class, but at this point, I cannot confirm that any of the remaining features won't be useful to the model.    

<h1><center>Model Selection</center></h1>
* K-Nearest Neighbors
* Suport Vector Classifier
* Logistic Regression
* Decision Tree Classifier
* Gaussian Naive Bayes 
* Random Forest Classifier
* Gradient Boosting Classifier   
  
I will revisit feature selection soon, but for now I will evaluate the base model performances of the seven classifiers above, and will move forward with the top two.  
  
**If you've made it this far, you deserve a little honesty..**  
Prior to this very moment, my only exposure to machine learning was roughly one week in mathematical statistics my junior year when we lightly covered linear regression. I'm sure there has to be better ways to go about selecting the best model for my problem, but I'm eager to learn, and I'm taking this project entirely as an opputtunity to learn.  
Plus, this way I'll have two different models to learn head-to-toe and that I know will at least perform semi-decent :-)
 
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

Although this dataset is from Kaggle, I'm not looking for a model that optimizes for accuracy at the expense of generality. I'm genuinely interested in the predictive capabilities of the 27 features, and would like my model to be able to predict on new data as well.  
For this to happen, I must estabalish a few precautionary measures.  
  
**High Variance**  
The dataset contains a less than preferrable number of samples, and because of this, the testing accuracy will be highly subject to variation. I want the two best performing models of the group, but I do not want them if they can't fit to new data. In attempt to mitigate this, I will use 10 repititions of 5-fold stratified cross validation when estimating each model's out-of-sample accuracy.  
  
**Data Leakage**  
Instead of passing the actual models to cross_val_score, I will cross validated pipelines containing both the model **and** the preprocessing steps. Thus, when the dataset is split into new training/test sets at each fold, the two sets will be preprocessed separately, and the chances of data leakage will be near-to-none!


```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

# seperate target variable from feature set
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

# computie and store each model's average CV accuracy
names = []
scores = []
for name, model in models:
    pipe = make_pipeline(preprocessor, model)
    rep_kfold = RepeatedStratifiedKFold(n_splits= 5, n_repeats= 10, random_state= 777)
    score = cross_val_score(pipe, X, Y, cv=rep_kfold, scoring='accuracy').mean() 
    
    names.append(name)
    scores.append(score)
    
# plot results
model_CVscores = pd.DataFrame({'Name': names, 'Score': scores})
model_CVscores.sort_values(by= 'Score', ascending= False).plot(kind='bar', 
                                                               x= 'Name', 
                                                               y= 'Score', 
                                                               style= 'ggplot', 
                                                               rot= 0, 
                                                               legend= False)

# print results
print(model_CVscores.sort_values(by= 'Score', ascending= False))
```
![](/images/math_ML_imgs/output_43_1.png)
  
![](/images/math_ML_imgs/model_scores.png)


Logistic Regession and Random Forest seem to be the best two classifiers for the dataset. Both scored over 65% accuracy right out of the box with default parameters and no feature selection.