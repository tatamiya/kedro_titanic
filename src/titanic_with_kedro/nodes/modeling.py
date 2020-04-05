import pandas as pd
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def model_construction(X, y, params):
    rfc = RandomForestClassifier(n_estimators=params['n_estimators'],
                                 random_state=params['random_state'])

    gs = GridSearchCV(estimator=rfc,
                      param_grid=params['param_grid'],
                      scoring=params['metrics'],
                      cv=params['n_cv'],
                      n_jobs=params['n_jobs'],
                      )
    gs.fit(X, y)

    logger = logging.getLogger(__name__)
    logger.info("Train Best Score is %.3f (%s)." %
                (gs.best_score_, params['metrics']))

    return gs


def prediction(df: pd.DataFrame, clf, df_org: pd.DataFrame):
    pred = clf.predict(df)

    df_pred = df_org.copy()
    df_pred['pred'] = pred

    return df_pred
