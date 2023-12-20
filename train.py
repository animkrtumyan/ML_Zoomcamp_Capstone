import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def train():
    output_file_dt = 'model_dt.bin'
    df= pd.read_csv('loan_approval_dataset.csv')
    df.columns = df.columns.str.strip()

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    df_train, df_val = train_test_split(df_full_train,
                                            test_size=0.25,
                                            random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_train.reset_index(drop=True)

    y_train = (df_train.loan_status == 'approved').astype(int).values
    y_val = (df_val.loan_status == 'approved').astype(int).values

    del df_train['loan_status']
    del df_val['loan_status']
    del df_test['loan_status']

    #print(df_full_train.shape, df_train.shape, df_val.shape, df_test.shape)

    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dict)

   #print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    dt = DecisionTreeClassifier(max_depth=15, min_samples_leaf=1)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_val)
    print("acc: ", accuracy_score(y_val, y_pred))


    with open(output_file_dt, 'wb') as f_out:
        pickle.dump((dv, dt), f_out)

    print("Model trained and saved successfully!")



if __name__ == '__main__':
    train()