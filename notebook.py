#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[227]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from IPython.display import display

from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score


from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.metrics import roc_auc_score
import pickle



# In[131]:


#!pip install scikit-learn==1.3.1


# In[132]:


# https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/code
data = pd.read_csv('loan_approval_dataset.csv')


# In[133]:


df= data.copy()


# # Data Processing

# In[134]:


df.head().T


# In[135]:


df.info()


# There are 13 features in the dataset, 13 numeric, 3 object type.
# the taret variable is loan_status, which is categorical and has two values:
# rejected or approved.
#  

# In[136]:


df.dtypes


# In[137]:


# As it is noticed in the above output, there are white spaces in the 
#colomn names,so string strip() method is used to remove white spaces.
df.columns = df.columns.str.strip()


# In[138]:


for c in df.columns:
    print(c)
    print(df[c].unique())


# In[139]:


df.describe()


# The statistics shows that are not very distinct outliers in the dataset, but this will be checked with visuals too.

# In[140]:


df.shape


# In[141]:


#checking missing values 
df.isnull().sum()
# There are no missing values


# In[142]:


df.duplicated().sum()
# no dublicates


# In[143]:


categorical_columns  = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower()
    df[c]  = df[c].str.strip()


# In[144]:


for c in df[categorical_columns]:
    print(c,"---------------------")
    print(df[c].value_counts(normalize = True))


# 
# There are categorical columns in the dataset:
# education
# [' Graduate' ' Not Graduate']
# self_employed
# [' No' ' Yes']
# loan_status
# [' Approved' ' Rejected']
# 

# In[145]:


df.info()


# # Train, test, validation splitting

# 
# The dataset was divided into  training, validation and testing sets.
# df_full_train was divided into  df_train (training set), df_val (validation set)
# 
# 

# In[146]:


df_full_train, df_test  = train_test_split(df, 
                                                test_size= 0.2,
                                                random_state = 1)


# In[147]:


dict(df_full_train[df_full_train.loan_id == 914])


# In[148]:


len(df_full_train), len(df_test) 


# In[149]:


# 0.25 of full train is 0.2 validation for the whole dataset
df_train, df_val  = train_test_split(df_full_train, 
                                     test_size= 0.25,
                                     random_state = 1)


# In[150]:


# reset indeces
df_train = df_train.reset_index(drop =  True)
df_val = df_train.reset_index(drop =  True)
df_test = df_test.reset_index(drop =  True)


# In[151]:


# get target variable churn
y_train  = (df_train.loan_status == 'approved').astype(int).values
y_val  = (df_val.loan_status == 'approved').astype(int).values
y_test  = (df_test.loan_status == 'approved').astype(int).values


# In[152]:


# delete target variable from training, validation, and test datasets
del df_train['loan_status']
del df_val['loan_status']
del df_test['loan_status']


# In[153]:


df_full_train.shape, df_train.shape, df_val.shape, df_test.shape


# # EDA

# In[154]:


# for EDA the full_train is used
df_full_train =  df_full_train.reset_index(drop = True)


# In[155]:


sns.countplot(x='loan_status', data= df_full_train)
plt.title("Distribution of the target variable-loan status")
plt.xlabel(" ")
plt.show()


# In[156]:


print("The approved loans are" ,round(df_full_train[df_full_train.loan_status == 'approved'].shape[0]/df_full_train.shape[0],2) *100, "%")
print("The rejected loans are" ,round(df_full_train[df_full_train.loan_status == 'rejected'].shape[0]/df_full_train.shape[0],2) *100, "%")


# In[157]:


for c in df_full_train[categorical_columns]:
    fig, axs = plt.subplots(figsize = (8,6))
    sns.countplot(x = c,
              hue ='loan_status', 
              data = df).set(
    title =  (f"Relationships of {c} and loan approval" ) ) 
plt.xlabel(c)


# The figures above represents distribution of the target variable "loan status" and categorical variables.
# Comparing education varable, it was noticed that both "graduate" and "not graduate" subgroups have almost the same level of loan approval.
# Similal pattern can be seen for "self-employment", so again, "yes" and "no" subgroups have almost the same laon approval.
# The last plot shows the distribution of target variable- loan status, where most of laons are approved.

# In[158]:


df_num = df_full_train.select_dtypes(include = 'number')
df_num = df_num.drop('loan_id', axis = 1)


# The following figures represent the distribution of numeric features.
# 1.no_of_dependences- this shows applicants' number of dependences, where the most frequent are 4 and 0 dependences.
# Income_annum- Annual income has many peaks $2M, $5M, $6M, $7M.
# The loan amount (requested amount) has a right-skewed distribution. The most frequently requested amount is $ 1.1M. 
# Loan term - the most frequent is 6 months.
# Cibil-score-the most frequent is 550.
# Residential and commercial asset values- have a right-skewed distribution with very long tales.
# Luxury and bank asset values have a right skewness, short tales
# Analysis of boxplots shows that almost every numeric feature does not have outliers. Only some values of asset values are a bit above the upper whisker.
# 
# 
# 

# In[159]:


for c in df_num:
    fig, axs = plt.subplots(figsize = (5,3))
    sns.histplot(df[c])
    plt.xlabel(c)
     


# In[160]:


for c in df_num:
    fig, axs = plt.subplots(figsize = (5,3))
    sns.boxplot(df[c])
    plt.xlabel(c)


# In[161]:


df_full_train.loan_status.value_counts()


# In[162]:


#number of unique values of categorical variables
df_full_train[categorical_columns].nunique()


# # Feature importance

# In[163]:


#using mutualinfo_score function for categorical variables


# In[164]:


def mutual_info_loan(series):
    return mutual_info_score(series, df_full_train.loan_status)


# In[165]:


selected_categorical_columns  = ['education', 'self_employed']


# In[166]:


mi = df_full_train[selected_categorical_columns].apply(mutual_info_loan)
mi.sort_values(ascending  = False)
# mi scores are low, so education and self-imploment are not informative for loan aproval


# In[167]:


df_full_train.loan_status = (df_full_train.loan_status == 'approved').astype(int).values


# In[168]:


df_full_train.loan_status


# In[169]:


# using correlation for numeric variables,
# a comperativley higher correleation shows cibil_score
df_num.corrwith(df_full_train.loan_status)


# In[170]:


dv = DictVectorizer(sparse = False)

train_dict = df_train.to_dict(orient = 'records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient = 'records')
X_val = dv.transform(val_dict)


# In[171]:


dv.get_feature_names_out()


# In[172]:


rf =  RandomForestClassifier()
rf.fit(X_train, y_train)


# In[173]:


# Feature importance with RandomForest Classifier
feature_importances = rf.feature_importances_
sorted_feature_importances = np.sort(feature_importances)
sorted_feature_names = np.array(dv.get_feature_names_out())[np.argsort(feature_importances)]

fig, axs = plt.subplots(figsize = (8,6))
plt.barh(y=sorted_feature_names, width =sorted_feature_importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Plot of loan approval')
plt.show()


# The most important features are 'cibil_score',  
# 'loan_term', 'loan_amount','income annum' according to RandomForest Classifier.
# The most frequent cibil score is 600-700, loan term - 6 months, as shows in the figures. 
# 

# In[174]:


fig, axs = plt.subplots(figsize = (8,6))
sns.histplot(df_full_train.cibil_score, bins = 10, shrink  = 0.8 )
plt.title("Distribution of cibil score")
plt.show()


# In[175]:


fig, axs = plt.subplots(figsize = (8,6))
sns.histplot(df_full_train.loan_term , shrink = 0.8 )
plt.title("Distribution of cibil score")
axs.set_xticks(df_full_train.loan_term.unique())
plt.show()


# In[176]:


df_full_train.loan_term.value_counts()


# In[177]:


fig, axs = plt.subplots(figsize = (8,6))
plt.scatter(x= df_full_train.income_annum, y=df_full_train.loan_amount )
plt.xlabel("yearly income ($)")
plt.ylabel("loan amou=t ($)")
plt.title("Relationship between income and loan amount")
plt.show()


# The scatter plot of yearly income and loan amount represents a linear relationship, as higher the income is, as higher the requested amount is.

# # Model training

# ## LogisticRegression

# In[178]:


model_log = LogisticRegression(max_iter=1000)
model_log.fit(X_train, y_train)
model_log.coef_# weights


# In[179]:


model_log.intercept_[0]


# In[180]:


print(model_log.score)


# In[181]:


model_log.predict(X_train)# predicting labels


# In[182]:


model_log.predict_proba(X_train)# predicting probabilities


# In[183]:


model_log.predict_proba(X_train)[:, 1]  


# In[184]:


y_pred = model_log.predict_proba(X_val)[:, 1]
y_pred 


# In[185]:


approve_decision =  (y_pred >= 0.5)
approve_decision


# In[186]:


# finding laons that are going to be approved
df_val[approve_decision].loan_id


# In[187]:


(y_val == approve_decision).mean().round(3)
# so the probability of loans to be acceped is abouut 75 %


# In[188]:


df_pred  = pd.DataFrame()
df_pred['loan_id'] = df_val.loan_id
df_pred['propability']= y_pred
df_pred['prediction']= approve_decision.astype(int)
df_pred['actual_values'] = y_val
df_pred['correct'] = (df_pred['prediction'] == df_pred['actual_values'])


# In[189]:


df_pred 


# In[190]:


df_full_train[df_full_train.loan_id == 3576 ]


# In[191]:


dicts_test = df_test.to_dict(orient = 'records')


# In[192]:


X_test = dv.transform(dicts_test )


# In[193]:


X_test


# In[194]:


y_pred  = model_log.predict_proba(X_test)[:, 1] 


# In[195]:


approve_decision =  (y_pred >= 0.5)
approve_decision


# In[196]:


(approve_decision == y_test).mean().round(3)


# In[197]:


customer  = dicts_test[5]
customer


# In[198]:


X_loan_id_574  = dv.transform([customer])
X_loan_id_574.shape


# In[199]:


model_log.predict_proba(X_loan_id_574 )[0,1]


# In[200]:


y_test[5]
# As it can be noticed, the model predicted of acception and
# the actual value was 1 == 'approved'


# In[201]:


df[df.loan_id == 574].loan_status


# In[202]:


df_pred_test  = pd.DataFrame()
df_pred_test['loan_id'] = df_test.loan_id
df_pred_test['propability']= y_pred
df_pred_test['prediction']= approve_decision.astype(int)
df_pred_test['actual_values'] = y_test
df_pred_test['correct'] = (df_pred_test['prediction'] == df_pred_test['actual_values'])


# In[208]:


df_pred_test[df_pred_test.prediction != df_pred_test.actual_values].shape[0]/df_pred_test.shape[0]


# So, about 24.7% are not correctly classified cases of the testing dataset. Here are the examples of those lead_ids.
# Therefore, let's try another model.

# In[209]:


df_pred_test[df_pred_test.prediction != df_pred_test.actual_values]


# ## DesisionTree

# In[210]:


for d in [1,2,3,4,5,6,7,8,9,10,15,20,25, None]:
    dt = DecisionTreeClassifier(max_depth = d)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    print('%4s -> %.3f' % (d, auc))


# In[211]:


scores = []
for d in [6,7,8,9,10,15]:
    for s in [1,2,5,10,15,20,100]:
        dt = DecisionTreeClassifier(max_depth = d, min_samples_leaf = s)
        dt.fit(X_train, y_train)
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d,s,auc))
    
          


# In[212]:


cols  = ["max_depth", "min_samples_leaf", "auc" ]
df_scores = pd.DataFrame(scores, columns  = cols)
df_scores.sort_values(by= "auc", ascending = False)


# In[213]:


df_scores_pivot = df_scores.pivot(index='min_samples_leaf',
                columns=["max_depth"],
                values=['auc'])

df_scores_pivot.round(3) 


# In[214]:


sns.heatmap(df_scores_pivot, annot = True, fmt = '.3f')


# In[215]:


# using best parameters
dt = DecisionTreeClassifier(max_depth = 15, min_samples_leaf = 1)
dt.fit(X_train, y_train)


# In[216]:


y_pred = dt.predict_proba(X_val)[:,1]


# In[217]:


print("auc: ",roc_auc_score(y_val, y_pred)) 
print("predicted probability is: ", rf.predict_proba(X_val[[0]]) )


# In[233]:


# let's make the prediction on the testing set: X_test
y_pred = dt.predict(X_test)


# In[234]:


accuracy = accuracy_score(y_test, y_pred)


# In[236]:


print("The accuracy score for the testing set is", round(accuracy,2)* 100, "%" )


# ## RandomForest Classifier

# In[263]:


rf = RandomForestClassifier(n_estimators= 10, random_state  = 1)
rf.fit(X_train, y_train)


# In[264]:


y_pred = rf.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred)


# In[258]:


rf.predict_proba(X_val[[0]])


# In[259]:


scores  = []
for n in range(10,201,10):
    rf = RandomForestClassifier(n_estimators= n, random_state  = 1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n, auc))


# In[242]:


df_scores = pd.DataFrame(scores, columns = ["n_estimators", "auc"])
df_scores


# In[243]:


plt.plot(df_scores.n_estimators, df_scores.auc)
plt.xlabel("number of trees (models)")
plt.ylabel('auc')
plt.show()


# In[244]:


n_estimators = 20


# In[245]:


for d in [1,2,3,4,5,6,7,8,9,10,15,20,25, None]:
    dt = RandomForestClassifier(max_depth = d,random_state=1,
                                n_estimators =n_estimators,
                                    n_jobs=-1)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    print('%4s -> %.3f' % (d, auc))


# In[246]:


scores  = []
for d in [5,6,7,8,9,10,15]:
    for n in range(10,201,10):
        rf = RandomForestClassifier(n_estimators= n,
                                    max_depth = d,
                                    random_state  = 1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d, n, auc))


# In[247]:


cols  =  ["max_depth", "n_estimators", "auc"]
df_scores = pd.DataFrame(scores, columns =cols )
df_scores


# In[248]:


for d in[6,7,8,9,10,15]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.auc,
            label = 'max_depth=%d' % d)
plt.legend()


# In[249]:


df_scores.head()


# In[250]:


max_depth = 10
scores = []
for s in [1,3, 5,10,50]:
    for n in range(10,201,10):
        rf = RandomForestClassifier(n_estimators= n,
                                    max_depth = max_depth,
                                    min_samples_leaf = s,
                                    random_state  = 1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((s, n, auc))


# In[251]:


cols  =  ["min_samples_leaf", "n_estimators", "auc"]
df_scores = pd.DataFrame(scores, columns =cols )
df_scores


# In[252]:


colors = ["blue", "orange","red","grey","black"]
min_samples_leaf = [1,3,5,10,50]
zip(min_samples_leaf, colors) # <zip at 0x23bc69fbdc0> 
# to show the values
list(zip(min_samples_leaf, colors) )


# In[253]:


for s, col in zip(min_samples_leaf, colors):
    df_subset = df_scores[df_scores.min_samples_leaf  == s]
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color = col,
            label = 'min_samples_leaf=%d' % s)
plt.legend()


# In[254]:


min_samples_leaf  = 1 


# Although Random Forest performs pretty well the hyperparameter tuning for getting better results.

# In[260]:


# the final model with best parameters
rf_final = RandomForestClassifier(n_estimators= 20,
                                    max_depth = max_depth,
                                  min_samples_leaf = min_samples_leaf, 
                                    random_state = 1,
                                    n_jobs = -1)
rf_final.fit(X_train, y_train)


# In[262]:


y_pred_fin= rf_final.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_fin)


#  The final model of RandomForest Classifier perfores well on testing data too.

# ## Xgboost

# In[265]:


features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = features)
dval = xgb.DMatrix(X_val, label = y_val, feature_names = features)
 


# In[266]:


# tuning eta

xgb_params = {
    'eta': 0.5,
    'max_depth': 10,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1,
    'verbosity': 2,
}

model  = xgb.train(xgb_params, dtrain, num_boost_round = 200)


# In[267]:


y_pred  = model.predict(dval)
y_pred


# In[268]:


roc_auc_score(y_val, y_pred)


# In[269]:


watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[270]:


scores ={}
#etas = ['eta=1.0', 'eta=0.7','eta=0.5']


# In[283]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta':0.5,\n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\n\nmodel  = xgb.train(xgb_params, dtrain,\n                   num_boost_round = 200,\n                   verbose_eval  = 5,\n                  evals = watchlist)\nmodel")


# In[284]:


def parse_xgb_output(output):
    results = []
    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line =  line.split('\t')
        
        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val  = float(val_line.split(':')[1])
        
        results.append((it, train, val))
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns = columns)
    return df_results


# In[285]:


df_scores  = parse_xgb_output(output)


# In[286]:


plt.plot(df_scores.num_iter, df_scores.train_auc, label = 'train')
plt.plot(df_scores.num_iter, df_scores.val_auc, label = 'val')
plt.legend()


# In[287]:


key = 'eta=%s' % (xgb_params['eta'])
scores[key] = parse_xgb_output(output)


# In[288]:


# key = 'eta=0.7', value is output
scores['eta=0.5'].head()


# In[289]:


scores.keys()


# In[290]:


key = 'eta=%s' % (xgb_params['eta'])
scores[key] = parse_xgb_output(output)
key


# In[291]:


for key, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label = key)
plt.legend()


# In[292]:


etas = ['eta=1.0', 'eta=0.7','eta=0.5']

for eta in etas:
    df_score = scores[eta]
    plt.plot(df_score.num_iter, df_score.val_auc, label = eta)
    
plt.ylim(0.9998,1.0001)
plt.legend()


# In[293]:


# the best eta is 0.7, though other values were good too


# In[295]:


scores ={}
# ['max_depth=6',7, 9, 10, 15]


# In[296]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1,\n    'max_depth':15,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'auc',\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n}\n\n\nmodel  = xgb.train(xgb_params, dtrain,\n                   num_boost_round = 200,\n                   verbose_eval  = 5,\n                  evals = watchlist)\nmodel")


# In[297]:


key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)
key


# In[298]:


df_scores  = parse_xgb_output(output)


# In[299]:


scores.keys()


# In[300]:


for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label = max_depth)
plt.ylim(0.998, 1.001)
plt.legend()


# In[302]:


# max _depth 9 shows the same auc curve as 10,15, so let's choose 9.


# In[303]:


# the final xgb model
xgb_params = {
    'eta': 0.7,
    'max_depth': 9,
      
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model_xgb_best  = xgb.train(xgb_params, dtrain,
                   num_boost_round = 175)
model_xgb_best 


# ## Comparing the models
# 

# In[305]:


y_pred  = model_log.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred)
 


# In[307]:


y_pred  = dt.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred)


# In[308]:


y_pred  = rf_final.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred)


# In[310]:


y_pred  = model_xgb_best.predict(dval)
roc_auc_score(y_val, y_pred)


# Based on auc values, it can be noticed that Desicion Tree Classifier 
#  and XGBoost Classifier perform better (with auc = 1.0) than
# Logistic regression (with auc = 0.84) and Random forest Classifier((with auc = 0.9924))for training faster this dataset, Decision Tree  Classifier was selected at the best model considering the computational power of local computers.

# # Saving the model 

# In[316]:


output_file_dt = 'model_dt.bin'


# In[317]:


with open(output_file_dt, 'wb') as f_out:
   pickle.dump((dv, dt), f_out)


# In[318]:


output_file_xg = 'model_xg.bin'


# In[319]:


with open(output_file_xg, 'wb') as f_out:
   pickle.dump((dv, model_xgb_best), f_out)


# In[ ]:





# In[ ]:




