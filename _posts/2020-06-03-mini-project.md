## Dataset Exploration

### Loading the Dataset


```python
%matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
```


```python
attrition_df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv") #dataset downloaded from https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset 
```

### Data Overview


```python
attrition_df.shape
```




    (1470, 35)




```python
pd.set_option("display.precision", 2, 'display.max_columns', None)
attrition_df.head(10)
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
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>Over18</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Female</td>
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>Sales Executive</td>
      <td>4</td>
      <td>Single</td>
      <td>5993</td>
      <td>19479</td>
      <td>8</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>Male</td>
      <td>61</td>
      <td>2</td>
      <td>2</td>
      <td>Research Scientist</td>
      <td>2</td>
      <td>Married</td>
      <td>5130</td>
      <td>24907</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>Male</td>
      <td>92</td>
      <td>2</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>2090</td>
      <td>2396</td>
      <td>6</td>
      <td>Y</td>
      <td>Yes</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>Female</td>
      <td>56</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>3</td>
      <td>Married</td>
      <td>2909</td>
      <td>23159</td>
      <td>1</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>Male</td>
      <td>40</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>2</td>
      <td>Married</td>
      <td>3468</td>
      <td>16632</td>
      <td>9</td>
      <td>Y</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <td>5</td>
      <td>32</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1005</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>Male</td>
      <td>79</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>4</td>
      <td>Single</td>
      <td>3068</td>
      <td>11864</td>
      <td>0</td>
      <td>Y</td>
      <td>No</td>
      <td>13</td>
      <td>3</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <td>6</td>
      <td>59</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1324</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>Female</td>
      <td>81</td>
      <td>4</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>1</td>
      <td>Married</td>
      <td>2670</td>
      <td>9964</td>
      <td>4</td>
      <td>Y</td>
      <td>Yes</td>
      <td>20</td>
      <td>4</td>
      <td>1</td>
      <td>80</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>30</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1358</td>
      <td>Research &amp; Development</td>
      <td>24</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>11</td>
      <td>4</td>
      <td>Male</td>
      <td>67</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Divorced</td>
      <td>2693</td>
      <td>13335</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>22</td>
      <td>4</td>
      <td>2</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>38</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>216</td>
      <td>Research &amp; Development</td>
      <td>23</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>12</td>
      <td>4</td>
      <td>Male</td>
      <td>44</td>
      <td>2</td>
      <td>3</td>
      <td>Manufacturing Director</td>
      <td>3</td>
      <td>Single</td>
      <td>9526</td>
      <td>8787</td>
      <td>0</td>
      <td>Y</td>
      <td>No</td>
      <td>21</td>
      <td>4</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <td>9</td>
      <td>36</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1299</td>
      <td>Research &amp; Development</td>
      <td>27</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>13</td>
      <td>3</td>
      <td>Male</td>
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>Healthcare Representative</td>
      <td>3</td>
      <td>Married</td>
      <td>5237</td>
      <td>16577</td>
      <td>6</td>
      <td>Y</td>
      <td>No</td>
      <td>13</td>
      <td>3</td>
      <td>2</td>
      <td>80</td>
      <td>2</td>
      <td>17</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
attrition_df.describe()
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
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobSatisfaction</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.0</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.0</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
      <td>1470.00</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>36.92</td>
      <td>802.49</td>
      <td>9.19</td>
      <td>2.91</td>
      <td>1.0</td>
      <td>1024.87</td>
      <td>2.72</td>
      <td>65.89</td>
      <td>2.73</td>
      <td>2.06</td>
      <td>2.73</td>
      <td>6502.93</td>
      <td>14313.10</td>
      <td>2.69</td>
      <td>15.21</td>
      <td>3.15</td>
      <td>2.71</td>
      <td>80.0</td>
      <td>0.79</td>
      <td>11.28</td>
      <td>2.80</td>
      <td>2.76</td>
      <td>7.01</td>
      <td>4.23</td>
      <td>2.19</td>
      <td>4.12</td>
    </tr>
    <tr>
      <td>std</td>
      <td>9.14</td>
      <td>403.51</td>
      <td>8.11</td>
      <td>1.02</td>
      <td>0.0</td>
      <td>602.02</td>
      <td>1.09</td>
      <td>20.33</td>
      <td>0.71</td>
      <td>1.11</td>
      <td>1.10</td>
      <td>4707.96</td>
      <td>7117.79</td>
      <td>2.50</td>
      <td>3.66</td>
      <td>0.36</td>
      <td>1.08</td>
      <td>0.0</td>
      <td>0.85</td>
      <td>7.78</td>
      <td>1.29</td>
      <td>0.71</td>
      <td>6.13</td>
      <td>3.62</td>
      <td>3.22</td>
      <td>3.57</td>
    </tr>
    <tr>
      <td>min</td>
      <td>18.00</td>
      <td>102.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>30.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1009.00</td>
      <td>2094.00</td>
      <td>0.00</td>
      <td>11.00</td>
      <td>3.00</td>
      <td>1.00</td>
      <td>80.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>30.00</td>
      <td>465.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>1.0</td>
      <td>491.25</td>
      <td>2.00</td>
      <td>48.00</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>2911.00</td>
      <td>8047.00</td>
      <td>1.00</td>
      <td>12.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>80.0</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>36.00</td>
      <td>802.00</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>1.0</td>
      <td>1020.50</td>
      <td>3.00</td>
      <td>66.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>4919.00</td>
      <td>14235.50</td>
      <td>2.00</td>
      <td>14.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>80.0</td>
      <td>1.00</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>1.00</td>
      <td>3.00</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>43.00</td>
      <td>1157.00</td>
      <td>14.00</td>
      <td>4.00</td>
      <td>1.0</td>
      <td>1555.75</td>
      <td>4.00</td>
      <td>83.75</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>8379.00</td>
      <td>20461.50</td>
      <td>4.00</td>
      <td>18.00</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>80.0</td>
      <td>1.00</td>
      <td>15.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>9.00</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>7.00</td>
    </tr>
    <tr>
      <td>max</td>
      <td>60.00</td>
      <td>1499.00</td>
      <td>29.00</td>
      <td>5.00</td>
      <td>1.0</td>
      <td>2068.00</td>
      <td>4.00</td>
      <td>100.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>19999.00</td>
      <td>26999.00</td>
      <td>9.00</td>
      <td>25.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>80.0</td>
      <td>3.00</td>
      <td>40.00</td>
      <td>6.00</td>
      <td>4.00</td>
      <td>40.00</td>
      <td>18.00</td>
      <td>15.00</td>
      <td>17.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
attrition_df['Attrition'].value_counts(normalize=True) #unbalanced dataset
```




    No     0.84
    Yes    0.16
    Name: Attrition, dtype: float64




```python
attrition_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
    Age                         1470 non-null int64
    Attrition                   1470 non-null object
    BusinessTravel              1470 non-null object
    DailyRate                   1470 non-null int64
    Department                  1470 non-null object
    DistanceFromHome            1470 non-null int64
    Education                   1470 non-null int64
    EducationField              1470 non-null object
    EmployeeCount               1470 non-null int64
    EmployeeNumber              1470 non-null int64
    EnvironmentSatisfaction     1470 non-null int64
    Gender                      1470 non-null object
    HourlyRate                  1470 non-null int64
    JobInvolvement              1470 non-null int64
    JobLevel                    1470 non-null int64
    JobRole                     1470 non-null object
    JobSatisfaction             1470 non-null int64
    MaritalStatus               1470 non-null object
    MonthlyIncome               1470 non-null int64
    MonthlyRate                 1470 non-null int64
    NumCompaniesWorked          1470 non-null int64
    Over18                      1470 non-null object
    OverTime                    1470 non-null object
    PercentSalaryHike           1470 non-null int64
    PerformanceRating           1470 non-null int64
    RelationshipSatisfaction    1470 non-null int64
    StandardHours               1470 non-null int64
    StockOptionLevel            1470 non-null int64
    TotalWorkingYears           1470 non-null int64
    TrainingTimesLastYear       1470 non-null int64
    WorkLifeBalance             1470 non-null int64
    YearsAtCompany              1470 non-null int64
    YearsInCurrentRole          1470 non-null int64
    YearsSinceLastPromotion     1470 non-null int64
    YearsWithCurrManager        1470 non-null int64
    dtypes: int64(26), object(9)
    memory usage: 402.1+ KB



```python
for c in attrition_df.iteritems():
    attribute = c[0]
    unique_val = attrition_df[attribute].unique()
    print(attribute) 
    print(unique_val)
```

    Age
    [41 49 37 33 27 32 59 30 38 36 35 29 31 34 28 22 53 24 21 42 44 46 39 43
     50 26 48 55 45 56 23 51 40 54 58 20 25 19 57 52 47 18 60]
    Attrition
    ['Yes' 'No']
    BusinessTravel
    ['Travel_Rarely' 'Travel_Frequently' 'Non-Travel']
    DailyRate
    [1102  279 1373 1392  591 1005 1324 1358  216 1299  809  153  670 1346
      103 1389  334 1123 1219  371  673 1218  419  391  699 1282 1125  691
      477  705  924 1459  125  895  813 1273  869  890  852 1141  464 1240
     1357  994  721 1360 1065  408 1211 1229  626 1434 1488 1097 1443  515
      853 1142  655 1115  427  653  989 1435 1223  836 1195 1339  664  318
     1225 1328 1082  548  132  746  776  193  397  945 1214  111  573 1153
     1400  541  432  288  669  530  632 1334  638 1093 1217 1353  120  682
      489  807  827  871  665 1040 1420  240 1280  534 1456  658  142 1127
     1031 1189 1354 1467  922  394 1312  750  441  684  249  841  147  528
      594  470  957  542  802 1355 1150 1329  959 1033 1316  364  438  689
      201 1427  857  933 1181 1395  662 1436  194  967 1496 1169 1145  630
      303 1256  440 1450 1452  465  702 1157  602 1480 1268  713  134  526
     1380  140  629 1356  328 1084  931  692 1069  313  894  556 1344  290
      138  926 1261  472 1002  878  905 1180  121 1136  635 1151  644 1045
      829 1242 1469  896  992 1052 1147 1396  663  119  979  319 1413  944
     1323  532  818  854 1034  771 1401 1431  976 1411 1300  252 1327  832
     1017 1199  504  505  916 1247  685  269 1416  833  307 1311  128  488
      529 1210 1463  675 1385 1403  452  666 1158  228  996  728 1315  322
     1479  797 1070  442  496 1372  920  688 1449 1117  636  506  444  950
      889  555  230 1232  566 1302  812 1476  218 1132 1105  906  849  390
      106 1249  192  553  117  185 1091  723 1220  588 1377 1018 1275  798
      672 1162  508 1482  559  210  928 1001  549 1124  738  570 1130 1192
      343  144 1296 1309  483  810  544 1062 1319  641 1332  756  845  593
     1171  350  921 1144  143 1046  575  156 1283  755  304 1178  329 1362
     1371  202  253  164 1107  759 1305  982  821 1381  480 1473  891 1063
      645 1490  317  422 1485 1368 1448  296 1398 1349  986 1099 1116 1499
      983 1009 1303 1274 1277  587  413 1276  988 1474  163  267  619  302
      443  828  561  426  232 1306 1094  509  775  195  258  471  799  956
      535 1495  446 1245  703  823 1246  622 1287  448  254 1365  538  525
      558  782  362 1236 1112  204 1343  604 1216  646  160  238 1397  306
      991  482 1176  913 1076  727  885  243  806  817 1410 1207 1442  693
      929  562  608  580  970 1179  294  314  316  654  168  381  217  501
      650  141  804  975 1090  346  430  268  167  621  527  883  954  310
      719  725  715  657 1146  182  376  571  384  791 1111 1243 1092 1325
      805  213  118  676 1252  286 1258  932 1041  859  720  946 1184  436
      589  760  887 1318  625  180  586 1012  661  930  342 1230 1271 1278
      607  130  300  583 1418 1269  379  395 1265 1222  341  868 1231  102
      881 1383 1075  374 1086  781  177  500 1425 1454  617 1085  995 1122
      618  546  462 1198 1272  154 1137 1188  188 1333  867  263  938  129
      616  498 1404 1053  289 1376  231  152  882  903 1379  335  722  461
      974 1126  840 1134  248  955  939 1391 1206  287 1441  109 1066  277
      466 1055  265  135  247 1035  266  145 1038 1234 1109 1089  788  124
      660 1186 1464  796  415  769 1003 1366  330 1492 1204  309 1330  469
      697 1262 1050  770  406  203 1308  984  439  793 1451 1182  174  490
      718  433  773  603  874  367  199  481  647 1384  902  819  862 1457
      977  942 1402 1421 1361  917  200  150  179  696  116  363  107 1465
      458 1212 1103  966 1010  326 1098  969 1167  694 1320  536  373  599
      251  131  237 1429  648  735  531  429  968  879  640  412  848  360
     1138  325 1322  299 1030  634  524  256 1060  935  495  282  206  943
      523  507  601  855 1291 1405 1369  999 1202  285  404  736 1498 1200
     1439  499  205  683 1462  949  652  332 1475  337  971 1174  667  560
      172  383 1255  359  401  377  592 1445 1221  866  981  447 1326  748
      990  405  115  790  830 1193 1423  467  271  410 1083  516  224  136
     1029  333 1440  674 1342  898  824  492  598  740  888 1288  104 1108
      479 1351  474  437  884 1370  264 1059  563  457 1313  241 1015  336
     1387  170  208  671  711  737 1470  365  763  567  486  772  301  311
      584  880  392  148  708 1259  786  370  678  146  581  918 1238  585
      741  552  369  717  543  964  792  611  176  897  600 1054  428  181
      211 1079  590  305  953  478 1375  244  511 1294  196  734 1239 1253
     1128 1336  234  766  261 1194  431  572 1422 1297  574  355  207  706
      280  726  414  352 1224  459 1254 1131  835 1172 1266  783  219 1213
     1096 1251 1394  605 1064 1337  937  157  754 1168  155 1444  189  911
     1321 1154  557  642  801  161 1382 1037  105  582  704  345 1120 1378
      468  613 1023  628]
    Department
    ['Sales' 'Research & Development' 'Human Resources']
    DistanceFromHome
    [ 1  8  2  3 24 23 27 16 15 26 19 21  5 11  9  7  6 10  4 25 12 18 29 22
     14 20 28 17 13]
    Education
    [2 1 4 3 5]
    EducationField
    ['Life Sciences' 'Other' 'Medical' 'Marketing' 'Technical Degree'
     'Human Resources']
    EmployeeCount
    [1]
    EmployeeNumber
    [   1    2    4 ... 2064 2065 2068]
    EnvironmentSatisfaction
    [2 3 4 1]
    Gender
    ['Female' 'Male']
    HourlyRate
    [ 94  61  92  56  40  79  81  67  44  84  49  31  93  50  51  80  96  78
      45  82  53  83  58  72  48  42  41  86  97  75  33  37  73  98  36  47
      71  30  43  99  59  95  57  76  87  66  55  32  52  70  62  64  63  60
     100  46  39  77  35  91  54  34  90  65  88  85  89  68  69  74  38]
    JobInvolvement
    [3 2 4 1]
    JobLevel
    [2 1 3 4 5]
    JobRole
    ['Sales Executive' 'Research Scientist' 'Laboratory Technician'
     'Manufacturing Director' 'Healthcare Representative' 'Manager'
     'Sales Representative' 'Research Director' 'Human Resources']
    JobSatisfaction
    [4 2 3 1]
    MaritalStatus
    ['Single' 'Married' 'Divorced']
    MonthlyIncome
    [5993 5130 2090 ... 9991 5390 4404]
    MonthlyRate
    [19479 24907  2396 ...  5174 13243 10228]
    NumCompaniesWorked
    [8 1 6 9 0 4 5 2 7 3]
    Over18
    ['Y']
    OverTime
    ['Yes' 'No']
    PercentSalaryHike
    [11 23 15 12 13 20 22 21 17 14 16 18 19 24 25]
    PerformanceRating
    [3 4]
    RelationshipSatisfaction
    [1 4 2 3]
    StandardHours
    [80]
    StockOptionLevel
    [0 1 3 2]
    TotalWorkingYears
    [ 8 10  7  6 12  1 17  5  3 31 13  0 26 24 22  9 19  2 23 14 15  4 29 28
     21 25 20 11 16 37 38 30 40 18 36 34 32 33 35 27]
    TrainingTimesLastYear
    [0 3 2 5 1 4 6]
    WorkLifeBalance
    [1 3 2 4]
    YearsAtCompany
    [ 6 10  0  8  2  7  1  9  5  4 25  3 12 14 22 15 27 21 17 11 13 37 16 20
     40 24 33 19 36 18 29 31 32 34 26 30 23]
    YearsInCurrentRole
    [ 4  7  0  2  5  9  8  3  6 13  1 15 14 16 11 10 12 18 17]
    YearsSinceLastPromotion
    [ 0  1  3  2  7  4  8  6  5 15  9 13 12 10 11 14]
    YearsWithCurrManager
    [ 5  7  0  2  6  8  3 11 17  1  4 12  9 10 15 13 16 14]


### Unknown Attribute Exploration


```python
payment_data = attrition_df[['MonthlyIncome','DailyRate','HourlyRate','MonthlyRate']]
payment_data.head()
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
      <th>MonthlyIncome</th>
      <th>DailyRate</th>
      <th>HourlyRate</th>
      <th>MonthlyRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5993</td>
      <td>1102</td>
      <td>94</td>
      <td>19479</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5130</td>
      <td>279</td>
      <td>61</td>
      <td>24907</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2090</td>
      <td>1373</td>
      <td>92</td>
      <td>2396</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2909</td>
      <td>1392</td>
      <td>56</td>
      <td>23159</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3468</td>
      <td>591</td>
      <td>40</td>
      <td>16632</td>
    </tr>
  </tbody>
</table>
</div>




```python
payment_data.corr()
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
      <th>MonthlyIncome</th>
      <th>DailyRate</th>
      <th>HourlyRate</th>
      <th>MonthlyRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MonthlyIncome</td>
      <td>1.00e+00</td>
      <td>7.71e-03</td>
      <td>-0.02</td>
      <td>0.03</td>
    </tr>
    <tr>
      <td>DailyRate</td>
      <td>7.71e-03</td>
      <td>1.00e+00</td>
      <td>0.02</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <td>HourlyRate</td>
      <td>-1.58e-02</td>
      <td>2.34e-02</td>
      <td>1.00</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <td>MonthlyRate</td>
      <td>3.48e-02</td>
      <td>-3.22e-02</td>
      <td>-0.02</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_div=payment_data['MonthlyRate']/payment_data['DailyRate']
data_div
```




    0       17.68
    1       89.27
    2        1.75
    3       16.64
    4       28.14
            ...  
    1465    13.90
    1466    35.00
    1467    33.38
    1468    12.95
    1469    16.29
    Length: 1470, dtype: float64




```python
data_div=payment_data['DailyRate']/payment_data['HourlyRate']
data_div
```




    0       11.72
    1        4.57
    2       14.92
    3       24.86
    4       14.78
            ...  
    1465    21.56
    1466    14.60
    1467     1.78
    1468    16.24
    1469     7.66
    Length: 1470, dtype: float64




```python
check_assumption = attrition_df[['Department','JobLevel', 'StockOptionLevel', 'MonthlyIncome']]
departments = ['Sales','Research & Development', 'Human Resources']
attribute_check = ['JobLevel','StockOptionLevel']

sns.set(rc={'figure.figsize':(15,10)})
sns.set_style("whitegrid")

for d in departments:
    for a in attribute_check:
        ax = sns.scatterplot(x=a, y="MonthlyIncome", data=check_assumption[check_assumption['Department']==d])
        ax.axes.set_title(f"{a} vs. Monthly Income for {d}",fontsize=15)
        ax.set_ylabel('Monthly Income',fontsize=15)
        ax.set_xlabel(a,fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()
```


![png](output_16_0.png)



![png](output_16_1.png)



![png](output_16_2.png)



![png](output_16_3.png)



![png](output_16_4.png)



![png](output_16_5.png)


### Attribute Removal


```python
attrition_df.drop(columns=['StockOptionLevel', 'DailyRate', 'HourlyRate', 'MonthlyRate', 'EmployeeCount', 'EmployeeNumber','Over18','StandardHours'],inplace=True)
attrition_df.head()
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
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>NumCompaniesWorked</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>Female</td>
      <td>3</td>
      <td>2</td>
      <td>Sales Executive</td>
      <td>4</td>
      <td>Single</td>
      <td>5993</td>
      <td>8</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>3</td>
      <td>Male</td>
      <td>2</td>
      <td>2</td>
      <td>Research Scientist</td>
      <td>2</td>
      <td>Married</td>
      <td>5130</td>
      <td>1</td>
      <td>No</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>4</td>
      <td>Male</td>
      <td>2</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>2090</td>
      <td>6</td>
      <td>Yes</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>4</td>
      <td>Female</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>3</td>
      <td>Married</td>
      <td>2909</td>
      <td>1</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>Male</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>2</td>
      <td>Married</td>
      <td>3468</td>
      <td>9</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Visualisation of Categorical Variables


```python
cat_variables = ['BusinessTravel','Department','Education','EducationField','EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','OverTime','PerformanceRating','RelationshipSatisfaction','WorkLifeBalance']

sns.set(rc={'figure.figsize':(15,10)})
sns.set_style("whitegrid")

for i in cat_variables: 
    ax = sns.countplot(x=i, hue="Attrition", data=attrition_df)
    ax.axes.set_title(f"Count of Employees Who Have Left and Stayed by {i}",fontsize=15)
    ax.set_xlabel(i,fontsize=15)
    ax.set_ylabel("Count of Employees",fontsize=15)
    plt.xticks(rotation=40)
    plt.setp(ax.get_legend().get_texts(), fontsize='15') 
    plt.setp(ax.get_legend().get_title(), fontsize='15') 
    plt.show()
    
    #https://github.com/mwaskom/seaborn/issues/1027:
    props = attrition_df.groupby(i)['Attrition'].value_counts(normalize=True).unstack()
    props.plot(kind='bar')
   
    plt.title(f"Proportion of Employees Who Have Left and Stayed by {i}",fontsize=15)
    plt.xlabel(i,fontsize=15)
    plt.xticks(rotation=40)
    plt.ylabel("Proportion of Employees",fontsize=15)
    plt.show()
```


![png](output_20_0.png)



![png](output_20_1.png)



![png](output_20_2.png)



![png](output_20_3.png)



![png](output_20_4.png)



![png](output_20_5.png)



![png](output_20_6.png)



![png](output_20_7.png)



![png](output_20_8.png)



![png](output_20_9.png)



![png](output_20_10.png)



![png](output_20_11.png)



![png](output_20_12.png)



![png](output_20_13.png)



![png](output_20_14.png)



![png](output_20_15.png)



![png](output_20_16.png)



![png](output_20_17.png)



![png](output_20_18.png)



![png](output_20_19.png)



![png](output_20_20.png)



![png](output_20_21.png)



![png](output_20_22.png)



![png](output_20_23.png)



![png](output_20_24.png)



![png](output_20_25.png)



![png](output_20_26.png)



![png](output_20_27.png)



![png](output_20_28.png)



![png](output_20_29.png)


### Visualisation of Numerical Variables


```python
num_variables = ['Age', 'DistanceFromHome','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'] 

sns.set(rc={'figure.figsize':(15,10)})
sns.set_style("whitegrid")

for i in num_variables:
    ax = sns.boxplot(x='Attrition', y=i, data=attrition_df)
    ax.axes.set_title(f"Distribution of Employees Who Have Left and Stayed by {i}",fontsize=15)
    ax.set_xlabel('Attrition',fontsize=15)
    ax.set_ylabel(i,fontsize=15)
    plt.show()
    
    mean_yes = attrition_df[attrition_df['Attrition']=='Yes'].mean()
    mean_no = attrition_df[attrition_df['Attrition']=='No'].mean()
    med_yes = attrition_df[attrition_df['Attrition']=='Yes'].median()
    med_no = attrition_df[attrition_df['Attrition']=='No'].median()
    max_yes = attrition_df[attrition_df['Attrition']=='Yes'].max()
    max_no = attrition_df[attrition_df['Attrition']=='No'].max()
    min_no = attrition_df[attrition_df['Attrition']=='No'].min()
    min_yes = attrition_df[attrition_df['Attrition']=='Yes'].min()
    q1_yes = attrition_df[attrition_df['Attrition']=='Yes'].quantile(0.25)
    q1_no = attrition_df[attrition_df['Attrition']=='No'].quantile(0.25)
    q3_yes = attrition_df[attrition_df['Attrition']=='Yes'].quantile(0.75)
    q3_no = attrition_df[attrition_df['Attrition']=='No'].quantile(0.75)
    
    print(f'employees who left (attrition=yes):')
    print(f'mean {i}: {mean_yes[i]:.2f}')
    print(f'median {i}: {med_yes[i]:.2f}')
    print(f'IQR {i}: {q3_yes[i]-q1_yes[i]:.2f}')
    print(f'min {i}: {min_yes[i]:.2f}')
    print(f'max {i}: {max_yes[i]:.2f}')
    print('\n')
    print(f'employees who stayed (attrition=no):')
    print(f'mean {i}: {mean_no[i]:.2f}')
    print(f'median {i}: {med_no[i]:.2f}')
    print(f'IQR {i}: {q3_no[i]-q1_no[i]:.2f}')
    print(f'min {i}: {min_no[i]:.2f}')
    print(f'max {i}: {max_no[i]:.2f}')
```


![png](output_22_0.png)


    employees who left (attrition=yes):
    mean Age: 33.61
    median Age: 32.00
    IQR Age: 11.00
    min Age: 18.00
    max Age: 58.00
    
    
    employees who stayed (attrition=no):
    mean Age: 37.56
    median Age: 36.00
    IQR Age: 12.00
    min Age: 18.00
    max Age: 60.00



![png](output_22_2.png)


    employees who left (attrition=yes):
    mean DistanceFromHome: 10.63
    median DistanceFromHome: 9.00
    IQR DistanceFromHome: 14.00
    min DistanceFromHome: 1.00
    max DistanceFromHome: 29.00
    
    
    employees who stayed (attrition=no):
    mean DistanceFromHome: 8.92
    median DistanceFromHome: 7.00
    IQR DistanceFromHome: 11.00
    min DistanceFromHome: 1.00
    max DistanceFromHome: 29.00



![png](output_22_4.png)


    employees who left (attrition=yes):
    mean MonthlyIncome: 4787.09
    median MonthlyIncome: 3202.00
    IQR MonthlyIncome: 3543.00
    min MonthlyIncome: 1009.00
    max MonthlyIncome: 19859.00
    
    
    employees who stayed (attrition=no):
    mean MonthlyIncome: 6832.74
    median MonthlyIncome: 5204.00
    IQR MonthlyIncome: 5623.00
    min MonthlyIncome: 1051.00
    max MonthlyIncome: 19999.00



![png](output_22_6.png)


    employees who left (attrition=yes):
    mean NumCompaniesWorked: 2.94
    median NumCompaniesWorked: 1.00
    IQR NumCompaniesWorked: 4.00
    min NumCompaniesWorked: 0.00
    max NumCompaniesWorked: 9.00
    
    
    employees who stayed (attrition=no):
    mean NumCompaniesWorked: 2.65
    median NumCompaniesWorked: 2.00
    IQR NumCompaniesWorked: 3.00
    min NumCompaniesWorked: 0.00
    max NumCompaniesWorked: 9.00



![png](output_22_8.png)


    employees who left (attrition=yes):
    mean PercentSalaryHike: 15.10
    median PercentSalaryHike: 14.00
    IQR PercentSalaryHike: 5.00
    min PercentSalaryHike: 11.00
    max PercentSalaryHike: 25.00
    
    
    employees who stayed (attrition=no):
    mean PercentSalaryHike: 15.23
    median PercentSalaryHike: 14.00
    IQR PercentSalaryHike: 6.00
    min PercentSalaryHike: 11.00
    max PercentSalaryHike: 25.00



![png](output_22_10.png)


    employees who left (attrition=yes):
    mean TotalWorkingYears: 8.24
    median TotalWorkingYears: 7.00
    IQR TotalWorkingYears: 7.00
    min TotalWorkingYears: 0.00
    max TotalWorkingYears: 40.00
    
    
    employees who stayed (attrition=no):
    mean TotalWorkingYears: 11.86
    median TotalWorkingYears: 10.00
    IQR TotalWorkingYears: 10.00
    min TotalWorkingYears: 0.00
    max TotalWorkingYears: 38.00



![png](output_22_12.png)


    employees who left (attrition=yes):
    mean TrainingTimesLastYear: 2.62
    median TrainingTimesLastYear: 2.00
    IQR TrainingTimesLastYear: 1.00
    min TrainingTimesLastYear: 0.00
    max TrainingTimesLastYear: 6.00
    
    
    employees who stayed (attrition=no):
    mean TrainingTimesLastYear: 2.83
    median TrainingTimesLastYear: 3.00
    IQR TrainingTimesLastYear: 1.00
    min TrainingTimesLastYear: 0.00
    max TrainingTimesLastYear: 6.00



![png](output_22_14.png)


    employees who left (attrition=yes):
    mean YearsAtCompany: 5.13
    median YearsAtCompany: 3.00
    IQR YearsAtCompany: 6.00
    min YearsAtCompany: 0.00
    max YearsAtCompany: 40.00
    
    
    employees who stayed (attrition=no):
    mean YearsAtCompany: 7.37
    median YearsAtCompany: 6.00
    IQR YearsAtCompany: 7.00
    min YearsAtCompany: 0.00
    max YearsAtCompany: 37.00



![png](output_22_16.png)


    employees who left (attrition=yes):
    mean YearsInCurrentRole: 2.90
    median YearsInCurrentRole: 2.00
    IQR YearsInCurrentRole: 4.00
    min YearsInCurrentRole: 0.00
    max YearsInCurrentRole: 15.00
    
    
    employees who stayed (attrition=no):
    mean YearsInCurrentRole: 4.48
    median YearsInCurrentRole: 3.00
    IQR YearsInCurrentRole: 5.00
    min YearsInCurrentRole: 0.00
    max YearsInCurrentRole: 18.00



![png](output_22_18.png)


    employees who left (attrition=yes):
    mean YearsSinceLastPromotion: 1.95
    median YearsSinceLastPromotion: 1.00
    IQR YearsSinceLastPromotion: 2.00
    min YearsSinceLastPromotion: 0.00
    max YearsSinceLastPromotion: 15.00
    
    
    employees who stayed (attrition=no):
    mean YearsSinceLastPromotion: 2.23
    median YearsSinceLastPromotion: 1.00
    IQR YearsSinceLastPromotion: 3.00
    min YearsSinceLastPromotion: 0.00
    max YearsSinceLastPromotion: 15.00



![png](output_22_20.png)


    employees who left (attrition=yes):
    mean YearsWithCurrManager: 2.85
    median YearsWithCurrManager: 2.00
    IQR YearsWithCurrManager: 5.00
    min YearsWithCurrManager: 0.00
    max YearsWithCurrManager: 14.00
    
    
    employees who stayed (attrition=no):
    mean YearsWithCurrManager: 4.37
    median YearsWithCurrManager: 3.00
    IQR YearsWithCurrManager: 5.00
    min YearsWithCurrManager: 0.00
    max YearsWithCurrManager: 17.00


### Replacing Non-numerical Data with Numerical Variables


```python
attrition_df['Attrition'].replace(['Yes'],1,inplace=True)
attrition_df['Attrition'].replace(['No'],0,inplace=True)
attrition_df.head() #Attrition replaced with 1s and 0s
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
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>NumCompaniesWorked</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>41</td>
      <td>1</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>Female</td>
      <td>3</td>
      <td>2</td>
      <td>Sales Executive</td>
      <td>4</td>
      <td>Single</td>
      <td>5993</td>
      <td>8</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>49</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>3</td>
      <td>Male</td>
      <td>2</td>
      <td>2</td>
      <td>Research Scientist</td>
      <td>2</td>
      <td>Married</td>
      <td>5130</td>
      <td>1</td>
      <td>No</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>37</td>
      <td>1</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>4</td>
      <td>Male</td>
      <td>2</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>2090</td>
      <td>6</td>
      <td>Yes</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>33</td>
      <td>0</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>4</td>
      <td>Female</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>3</td>
      <td>Married</td>
      <td>2909</td>
      <td>1</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>27</td>
      <td>0</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>Male</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>2</td>
      <td>Married</td>
      <td>3468</td>
      <td>9</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
attrition_df.info() #checking to see which variables we need to convert, we need to convert all the variables with the datatype object 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 27 columns):
    Age                         1470 non-null int64
    Attrition                   1470 non-null int64
    BusinessTravel              1470 non-null object
    Department                  1470 non-null object
    DistanceFromHome            1470 non-null int64
    Education                   1470 non-null int64
    EducationField              1470 non-null object
    EnvironmentSatisfaction     1470 non-null int64
    Gender                      1470 non-null object
    JobInvolvement              1470 non-null int64
    JobLevel                    1470 non-null int64
    JobRole                     1470 non-null object
    JobSatisfaction             1470 non-null int64
    MaritalStatus               1470 non-null object
    MonthlyIncome               1470 non-null int64
    NumCompaniesWorked          1470 non-null int64
    OverTime                    1470 non-null object
    PercentSalaryHike           1470 non-null int64
    PerformanceRating           1470 non-null int64
    RelationshipSatisfaction    1470 non-null int64
    TotalWorkingYears           1470 non-null int64
    TrainingTimesLastYear       1470 non-null int64
    WorkLifeBalance             1470 non-null int64
    YearsAtCompany              1470 non-null int64
    YearsInCurrentRole          1470 non-null int64
    YearsSinceLastPromotion     1470 non-null int64
    YearsWithCurrManager        1470 non-null int64
    dtypes: int64(20), object(7)
    memory usage: 310.2+ KB



```python
object_vars = attrition_df.select_dtypes(include=['object'])

for c in object_vars.iteritems():
    attribute = c[0]
    unique_val = attrition_df[attribute].unique()
    print(attribute) 
    print(unique_val)
```

    BusinessTravel
    ['Travel_Rarely' 'Travel_Frequently' 'Non-Travel']
    Department
    ['Sales' 'Research & Development' 'Human Resources']
    EducationField
    ['Life Sciences' 'Other' 'Medical' 'Marketing' 'Technical Degree'
     'Human Resources']
    Gender
    ['Female' 'Male']
    JobRole
    ['Sales Executive' 'Research Scientist' 'Laboratory Technician'
     'Manufacturing Director' 'Healthcare Representative' 'Manager'
     'Sales Representative' 'Research Director' 'Human Resources']
    MaritalStatus
    ['Single' 'Married' 'Divorced']
    OverTime
    ['Yes' 'No']



```python
# i'm going to explicitly set the numerical values for each of these categorical variables, so I know know each value represents 
new_cat_values = {'BusinessTravel': {'Travel_Rarely': 0, 'Travel_Frequently': 1,'Non-Travel':2 },
                  'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
                  'EducationField': {'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5}, 
                  'Gender' : {'Female':0 ,'Male':1},
                  'JobRole' : {'Sales Executive': 0, 'Research Scientist' :1, 'Laboratory Technician' :2, 'Manufacturing Director':3, 'Healthcare Representative' : 4,
                               'Manager': 5, 'Sales Representative' : 6, 'Research Director' : 7, 'Human Resources': 8},
                  'MaritalStatus' : {'Single':0, 'Married':1, 'Divorced':2},
                  'OverTime' : {'Yes':1, 'No':0}}      
```


```python
attrition_df_new = attrition_df.copy()
attrition_df_new.replace(new_cat_values, inplace=True)
attrition_df_new.head()
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
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>NumCompaniesWorked</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>41</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5993</td>
      <td>8</td>
      <td>1</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <td>1</td>
      <td>49</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5130</td>
      <td>1</td>
      <td>0</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>2</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>2090</td>
      <td>6</td>
      <td>1</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2909</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3468</td>
      <td>9</td>
      <td>0</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
attrition_df_new.info() #all values are now numerical 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 27 columns):
    Age                         1470 non-null int64
    Attrition                   1470 non-null int64
    BusinessTravel              1470 non-null int64
    Department                  1470 non-null int64
    DistanceFromHome            1470 non-null int64
    Education                   1470 non-null int64
    EducationField              1470 non-null int64
    EnvironmentSatisfaction     1470 non-null int64
    Gender                      1470 non-null int64
    JobInvolvement              1470 non-null int64
    JobLevel                    1470 non-null int64
    JobRole                     1470 non-null int64
    JobSatisfaction             1470 non-null int64
    MaritalStatus               1470 non-null int64
    MonthlyIncome               1470 non-null int64
    NumCompaniesWorked          1470 non-null int64
    OverTime                    1470 non-null int64
    PercentSalaryHike           1470 non-null int64
    PerformanceRating           1470 non-null int64
    RelationshipSatisfaction    1470 non-null int64
    TotalWorkingYears           1470 non-null int64
    TrainingTimesLastYear       1470 non-null int64
    WorkLifeBalance             1470 non-null int64
    YearsAtCompany              1470 non-null int64
    YearsInCurrentRole          1470 non-null int64
    YearsSinceLastPromotion     1470 non-null int64
    YearsWithCurrManager        1470 non-null int64
    dtypes: int64(27)
    memory usage: 310.2 KB


### Correlation


```python
correlation = attrition_df_new.corr(method='pearson') #correlation coefficient
correlation_all = abs(correlation['Attrition'])
correlation_all_sorted = correlation_all.sort_values(ascending=False)
correlation_all_sorted.drop('Attrition',inplace=True)
print(correlation_all_sorted)
print('\n')
print('top 10 correlated variables by absolute value of the correlation coefficient')
print(correlation_all_sorted.head(10))
```

    OverTime                    2.46e-01
    TotalWorkingYears           1.71e-01
    JobLevel                    1.69e-01
    MaritalStatus               1.62e-01
    YearsInCurrentRole          1.61e-01
    MonthlyIncome               1.60e-01
    Age                         1.59e-01
    YearsWithCurrManager        1.56e-01
    YearsAtCompany              1.34e-01
    JobInvolvement              1.30e-01
    JobSatisfaction             1.03e-01
    EnvironmentSatisfaction     1.03e-01
    DistanceFromHome            7.79e-02
    EducationField              7.54e-02
    Department                  6.40e-02
    WorkLifeBalance             6.39e-02
    TrainingTimesLastYear       5.95e-02
    RelationshipSatisfaction    4.59e-02
    NumCompaniesWorked          4.35e-02
    YearsSinceLastPromotion     3.30e-02
    Education                   3.14e-02
    Gender                      2.95e-02
    JobRole                     2.79e-02
    PercentSalaryHike           1.35e-02
    PerformanceRating           2.89e-03
    BusinessTravel              7.38e-05
    Name: Attrition, dtype: float64
    
    
    top 10 correlated variables by absolute value of the correlation coefficient
    OverTime                0.25
    TotalWorkingYears       0.17
    JobLevel                0.17
    MaritalStatus           0.16
    YearsInCurrentRole      0.16
    MonthlyIncome           0.16
    Age                     0.16
    YearsWithCurrManager    0.16
    YearsAtCompany          0.13
    JobInvolvement          0.13
    Name: Attrition, dtype: float64



```python
# code for heatmap retrieved from: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
corr = attrition_df_new.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```


![png](output_32_0.png)

