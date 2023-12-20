**Problem description**
 

The **dataset** refers to the Loan Approval Prediction problem and it was downloaded from the Kaggle [datasets](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/code). 

The **purpose** of the project is to predict loan approvment based on given critera. There are 4269 entities in the dataset. 
The total number of columns are 13, where 10 are numeric and 3 are categorical  features. 
The feature (columns) are the following:
 loan_id                       int64 -  unique ids for each loan,
 no_of_dependents             int64 - number of dependencies from the interval of 0 to 5 integers.  
 education                   object - being 'graduated' or 'not graduated'.
 self_employed               object - 2 category- "yes" and "no".
 income_annum                 int64 - the annual income.
 loan_amount                  int64 - requested amount.
 loan_term                    int64 -  a time for paying back the loan by month [12,  8, 20, 10,  4,  2, 18, 16, 14,  6]
 cibil_score                  int64 - [CIBIL Score](https://www.hdfcbank.com/personal/resources/learning-centre/borrow/what-is-the-cibil-credit-score-and-why-should-it-matter#:~:text=CIBIL%20Score%20is%20a%203,better%20your%20credit%20rating%20is.) is a 3-digit numeric summary of your credit history, rating and report, and ranges from 300 to 900.
 residential_assets_value     int64 - the value of the property
 commercial_assets_value      int64 - commertial property
 luxury_assets_value          int64 - luxury assets value 
 bank_asset_value             int64 - bank asset value
 loan_status                 object - "rejected" or "approved"

The problem of loan approval is an important issue in financial services where loans are given to those applicant 
who are not able to return the loan.So, it is vital to predict the possibility of returning financial resources and approve only those applicants
who are ready to pay back the money on time.

As _objectives_ of this project were selected:
-identify important features,
-conduct EDA for finding out patterns and correlations of the data features,
-use various types of modeling for prediction of loans, that worth to be approved,
-use trained models for further analysis and predictions.

The results of the project can be used for designing models to solving loan approval tasks. For example, given a new aplicant and usng the models, 
it could be possible to identify the possibility of returning the money and, so the model will help to make a decision approve 
the loan or reject it.

**Problem solution**
For solving loan approval problem, there were used predictions based on various models: Logistic regression, Decision Tree Classifier,
RandomForest Classifier, Xgboost classifier. Hyperparameter tuning was implemented for the models. All models perform more than 84% accuracy score.Considering auc values,Desicion Tree Classifier and XGBoost Classifier perform better
(with auc = 1.0) than Logistic regression (with auc = 0.84) and Random forest Classifier((with auc = 0.9924)) 
for training faster this dataset, Decision Tree Classifier was selected at the best model considering the computational power of local computers.


**Virtual environment**
For activating virtual environment, I used "python -m waitress --listen=*:9696 predict:app" command in the terminal. Parallely, the jupyter notebook was used to send POST requests.

For creating virtual envrinment  pipnev was used. The all versions of necessary libraries were taken via pip freeze command, that outputs all libraries and their versions.
After, I selected those liraries that are used for the Midterm project. This list was savd in the requirements.txt. Later, the requirements.txt was used for creating virtual environment.
There was a mismatch in the dependences, so  used pipenv lock --pre command, to find out the issue.
pipenv install -r requirements.txt
If this will not work, just try the bellow command in the terminal (cmd Windows).
pipenv install numpy pandas requests scikit-learn seaborn  waitress xgboost matplotlib  flask .
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.

**Docker**



