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


def preprocess_numerical(df: pd.DataFrame) -> (np.array, Pipeline):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    prep_numerical = Pipeline([('imputer', imputer),
                               ('scaler', scaler)])
    prepped_numerical = prep_numerical.fit_transform(df)
    df_prepped_num = pd.DataFrame(prepped_numerical, columns=df.columns)

    return df_prepped_num, prep_numerical


def preprocess_categorical(df: pd.DataFrame) -> (np.array, Pipeline):
    imputer = SimpleImputer(strategy='constant')
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    prep_categorical = Pipeline([('imputer', imputer),
                                 ('encoder', encoder)])
    prepped_categorical = prep_categorical.fit_transform(df)

    columns_prepped = prep_categorical.named_steps['encoder'].get_feature_names(df.columns)
    df_prepped_cat = pd.DataFrame(prepped_categorical, columns=columns_prepped)

    return df_prepped_cat, prep_categorical


def concat_prepped(prepped_num: pd.DataFrame, prepped_cat: pd.DataFrame) -> pd.DataFrame:
    prepped_concat = pd.concat([prepped_num, prepped_cat], axis=1)
    return prepped_concat
