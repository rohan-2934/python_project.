#!/usr/bin/env python
# coding: utf-8

# Problem Statement: Understand how the bank decides whether or not to grant a loan. 
# To assist the bank in reducing credit and interest risk, identify different patterns and reflect the outcomes.
# 
# The two input files are extracted, cleaned/transformed, and a few columns are examined using various charts created with Python packages. 
# The results are then used to draw certain conclusions.

# # importing the libraries and files

# In[5]:


import pandas as pd,numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[8]:


inp1=pd.read_csv(r"D:\Bank Loan Analysis data.csv")
inp1.head()


# In[9]:


inp1.sample() ##random sample


# # Application data sanity check

# In[9]:


inp1.head()


# In[4]:


inp1.shape ##no of rows and columns 


# In[5]:


type(inp1) ##2-d data


# In[12]:


inp1.info() ##gives the total information..


# In[13]:


inp1.info(verbose=True,null_counts =  True)


# In[7]:


inp1.describe()  ## it gives the statistical data , works only for numerical data , remianing columns are text data..


# In[16]:


inp1["AMT_ANNUITY"].median() ##50% (gives median)  (percentile)


# In[20]:


inp1["AMT_ANNUITY"].quantile(0.75) ##75% percentile..


# ## Data Analysis For Application Data

# ### checking the Application Dataset

# In[22]:


inp1.isnull()


# In[23]:


inp1.isnull().sum() ##it gives total null count on column basis.. 


# In[9]:


round(inp1.isnull().mean()*100,2).sort_values(ascending =  False)  ##percentage of null values..


# In[28]:


(inp1.isnull().sum()*100/len(inp1)).sort_values(ascending =  False) ##percentage of null values..


# In[39]:


inp1.loc[0:3,"SK_ID_CURR"]  ##label based indexing..(location based indexing)..(including both start and stop values)


# In[36]:


inp1.iloc[0:3,0:3]  ##integer based location..(excluded stop value)


# In[31]:


inp1.head()


# In[40]:


# removing all the columns with more than 50% null values
inp1 = inp1.loc[:,inp1.isnull().mean()<=0.5]
inp1.shape


# In[11]:


# taking columns with less or equal to than 13 % null values
list(inp1.columns[(inp1.isnull().mean()<=0.13) & (inp1.isnull().mean()>0)])


# In[ ]:





# In[ ]:





# ### Checking for values to impute in columns
# #### imputation for  EXT_SOURCE_2 

# In[12]:


inp1["EXT_SOURCE_2"].value_counts()


# In[13]:


plt.style.use("ggplot")
plt.figure(figsize = [12,6])
sns.boxplot(inp1["EXT_SOURCE_2"])
plt.show()


# In[41]:


sns.boxplot(inp1["EXT_SOURCE_2"])
plt.show()


# In[14]:


## because EXT Source 2 contains no outliers, we may impute the column using mean
imputeValue = round(inp1["EXT_SOURCE_2"].mean(),2)
imputeValue


# In[45]:


inp1["EXT_SOURCE_2"].quantile(0.75)


# In[15]:


#imputation for Occupation type


# In[16]:


inp1["AMT_ANNUITY"].value_counts()


# In[17]:


plt.style.use("ggplot")
plt.figure(figsize = [30,10])
sns.boxplot(inp1["AMT_ANNUITY"])
plt.show()


# In[46]:


sns.boxplot(inp1["AMT_ANNUITY"])
plt.show()


# In[18]:


# because AMT ANNUITY  contains outliers,the column can be imputed using the columns median
imputeVALUE = round(inp1["AMT_ANNUITY"].median(),2)
imputeVALUE


# # imputation for NAME_TYPE_SUITE

# In[19]:


inp1["NAME_TYPE_SUITE"].value_counts()


# In[20]:


inp1["NAME_TYPE_SUITE"]


# ### The column NAME_TYPE_SUITE is clearly a categorical(objec data type or text data type) one. As a result, the mode of the column can be used to impute this column.

# In[21]:


imputeVALUE = inp1["NAME_TYPE_SUITE"].mode()
imputeVALUE


# ### imputation for CNT_FAM_MEMBERS

# In[22]:


sns.boxplot(inp1["CNT_FAM_MEMBERS"])
plt.show()


# In[23]:


# we have outliers, the median is the good technique for imputation
imputeVALUE = round(inp1["CNT_FAM_MEMBERS"].median(),2)
imputeVALUE


# ### imputation for AMT_GOODS_PRICE

# In[24]:


inp1["AMT_GOODS_PRICE"].value_counts()


# In[25]:


sns.boxplot(inp1["AMT_GOODS_PRICE"])
plt.show()


# In[26]:


# we have outliers, the median is the good technique for imputation
imputeVALUE = round(inp1["AMT_GOODS_PRICE"].median(),2)
imputeVALUE


# ## Checking datatypes of columns and modify them appropriately

# In[28]:


# checking the columns of float type
inp1.select_dtypes(include = "float64").columns


# In[51]:


x=2
y=5
z = lambda x, y: x + y
z(x,y)


# In[29]:


#convert these columns to int type
Coltoconvert = ["OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","DAYS_LAST_PHONE_CHANGE",
               "AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON",
               "AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR"]

inp1.loc[:,Coltoconvert]= inp1.loc[:,Coltoconvert].apply(lambda col: col.astype("int",errors = "ignore"))


# In[31]:


## checking the columns having object type
Coltoconvert = list(inp1.select_dtypes(include = "object").columns)
Coltoconvert


# In[32]:


inp1.loc[:,Coltoconvert]= inp1.loc[:,Coltoconvert].apply(lambda col: col.astype("str",errors = "ignore"))


# In[33]:


inp1


# In[34]:


inp1.info(verbose = True,null_counts =  True)


# In[35]:


# FINDING DIFFERENT GENDERS FOR LOAN APPLICATION
inp1["CODE_GENDER"].value_counts()


# In[55]:


# Dropping the gender = we can drop the transgender as we have very limited data to analyze
inp1 = inp1[inp1["CODE_GENDER"]!= "XNA"]
inp1["CODE_GENDER"].replace(["M","F"],["MALE","Female"],inplace = True)
inp1


# In[53]:


inp1[(inp1["CODE_GENDER"]=='M')]


# In[40]:


# Binning Varaiables for analysis
inp1["AMT_INCOME_TOTAL"].quantile([0,0.1,0.3,0.6,0.8,1])  ### quantiles are user defined and not mandatory to set the fix values


# In[47]:


## creating a new categorical variable based on total income, binlabels should be always -1 to the quantile size
inp1["INCOME_GROUP"]=pd.qcut(inp1["AMT_INCOME_TOTAL"],q= [0.0,0.1,0.3,0.6,0.8,1],labels=["VeryLow","Low","Medium","High","VeryHigh"])


# In[56]:


# creating a column AGE using DAYS_Birth
inp1["AGE"]= abs(inp1["DAYS_BIRTH"])//365
inp1


# In[49]:


inp1["AGE"].describe()


# In[50]:


## because the age ranges from 20 to 69,we can divide it into five_year bins beginning at 20 and ending at 70
inp1["AGE_GROUP"] = pd.cut(inp1["AGE"],bins = np.arange(20,71,5))


# In[52]:


### Adding a new column 
inp1["CREDIT_INCOME_RATIO"]=round((inp1["AMT_CREDIT"]/inp1["AMT_INCOME_TOTAL"]))


# In[54]:


### inspecting targer for imbalance
inp1["TARGET"].value_counts(normalize = True)*100


# In[59]:


plt.pie(inp1["TARGET"].value_counts(normalize = True)*100)
plt.show()


# In[59]:


plt.boxplot(inp1["AMT_GOODS_PRICE"])
plt.show()


# In[ ]:


##EDA stands for Exploratory Data Analysis, which is a critical step in the data analysis process. It involves understanding the main characteristics of a dataset, typically by visualizing patterns, detecting anomalies, and summarizing its key aspects using statistical graphics and numerical methods.

Key Objectives of EDA:
Understanding the Data Structure: Identify variables (features) and their types (numerical, categorical, etc.).
Data Cleaning: Detect missing data, handle duplicates, correct errors, and deal with outliers.
Statistical Summary: Compute central tendencies (mean, median, mode), variability (variance, standard deviation), and distribution characteristics.
Data Visualization: Use plots and charts (e.g., histograms, scatter plots, box plots) to visualize relationships, distributions, and trends.
Common Techniques Used in EDA:
Summary Statistics: Mean, median, mode, standard deviation, quartiles, etc.
Data Visualizations:
Histograms: To check the distribution of numerical data.
Box Plots: To visualize the spread of data and detect outliers.
Scatter Plots: To explore relationships between two variables.
Heatmaps: To visualize correlations between variables.
Bar Charts: To summarize categorical data.
Importance of EDA:
Prepares data for modeling by identifying patterns, relationships, and anomalies.
Guides feature selection by highlighting key variables and relationships.
Improves model performance by providing insights into data transformations and cleaning.
EDA is typically done before formal modeling to ensure the data is well-understood and ready for further analysis. It’s a mix of visual and statistical exploration to make sense of the data quickly and efficiently.

get_ipython().run_line_magic('pinfo', 'EDA')

