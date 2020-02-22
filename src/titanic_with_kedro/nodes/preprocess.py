import pandas as pd
from sklearn import preprocessing


def _label_encoding(df: pd.DataFrame) -> (pd.DataFrame, dict):
    
    df_le = df.copy()
    # Getting Dummies from all categorical vars
    list_columns_object = df_le.columns[df_le.dtypes == 'object']
    
    dict_encoders = {}
    for column in list_columns_object:    
        le = preprocessing.LabelEncoder()
        mask_nan = df_le[column].isnull()
        df_le[column] = le.fit_transform(df_le[column].fillna('NaN'))
        
        df_le.loc[mask_nan, column] *= -1  # transform minus for missing columns
        dict_encoders[column] = le
    
    return df_le, dict_encoders


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    
    df_prep = df.copy()
    
    drop_cols = ['Name', 'Ticket', 'PassengerId']
    df_prep = df_prep.drop(drop_cols, axis=1)
    
    df_prep['Age'] = df_prep['Age'].fillna(df_prep['Age'].mean())

    # Filling missing Embarked values with most common value
    df_prep['Embarked'] = df_prep['Embarked'].fillna(df_prep['Embarked'].mode()[0])

    df_prep['Pclass'] = df_prep['Pclass'].astype(str)

    # Take the frist alphabet from Cabin
    df_prep['Cabin'] = df_prep['Cabin'].str[0]

    # Label Encoding for str columns
    df_prep, _ = _label_encoding(df_prep)
    
    return df_prep
