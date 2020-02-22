from typing import Dict, List
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    """Splits data into training and test sets.

        Args:
            data: Source data.
            parameters: Parameters defined in parameters.yml.

        Returns:
            A list containing split data.

    """
    target_col = 'Survived'
    X = data.drop(target_col, axis=1).values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test]


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train the random forest classifier model.

        Args:
            X_train: Training data of independent features.
            y_train: Training data for Survived.

        Returns:
            Trained model.

    """
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray):
    """Calculate the F1 score and log the result.

        Args:
            clf: Trained model.
            X_test: Testing data of independent features.
            y_test: Testing data for price.

    """
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has an F1 score of %.3f.", score)
