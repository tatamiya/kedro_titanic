import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


COLUMN_TARGET = 'Survived'
COLUMNS_NUMERICAL = ['Age', 'Fare']
COLUMNS_CATEGORICAL = ['Pclass', 'Sex', 'Embarked',
                       'SibSp', 'Parch', 'CabinInitialAlphabet']


def feature_target_split(df: pd.DataFrame):
    df_features = df.drop(COLUMN_TARGET, axis=1)
    y = df[COLUMN_TARGET]

    return df_features, y


def take_initial_character(df: pd.DataFrame) -> pd.DataFrame:
    df['CabinInitialAlphabet'] = df['Cabin'].str[0]
    return df


def split_numerical_categorical(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_numerical = df[COLUMNS_NUMERICAL]
    df_categorical = df[COLUMNS_CATEGORICAL]

    return df_numerical, df_categorical


def preprocess_numerical_train(df: pd.DataFrame) -> (np.array, Pipeline):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    prep = Pipeline([('imputer', imputer),
                     ('scaler', scaler)])
    prepped = prep.fit_transform(df)
    df_prepped = pd.DataFrame(prepped, columns=df.columns)

    return df_prepped, prep


def preprocess_numerical_test(df: pd.DataFrame, prep: Pipeline):
    prepped = prep.transform(df)
    df_prepped = pd.DataFrame(prepped, columns=df.columns)

    return df_prepped


def preprocess_categorical_train(df: pd.DataFrame) -> (np.array, Pipeline):
    imputer = SimpleImputer(strategy='constant')
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    prep = Pipeline([('imputer', imputer),
                     ('encoder', encoder)])
    prepped = prep.fit_transform(df)

    columns_prepped = prep.named_steps['encoder'].get_feature_names(df.columns)
    df_prepped = pd.DataFrame(prepped, columns=columns_prepped)

    return df_prepped, prep


def preprocess_categorical_test(df: pd.DataFrame, prep: Pipeline):
    prepped = prep.transform(df)
    columns_prepped = prep.named_steps['encoder'].get_feature_names(df.columns)
    df_prepped = pd.DataFrame(prepped, columns=columns_prepped)
    
    return df_prepped


def concat_prepped(prepped_num: pd.DataFrame, prepped_cat: pd.DataFrame) -> pd.DataFrame:
    prepped_concat = pd.concat([prepped_num, prepped_cat], axis=1)
    return prepped_concat


def validate_columns_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame):
    columns_train = df_train.columns
    columns_test = df_test.columns
    assert (columns_train == columns_test).all(), 'The columns are inconsistent with those of train data!'
