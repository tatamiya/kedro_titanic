from kedro.pipeline import node, Pipeline
from titanic_with_kedro.nodes import modeling


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=modeling.split_data,
                inputs=["train_prep", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),
            node(func=modeling.train_model,
                 inputs=["X_train", "y_train"],
                 outputs="clf"),
            node(
                func=modeling.evaluate_model,
                inputs=["clf", "X_test", "y_test"],
                outputs=None,
            ),
        ],
        tags=["ds_tag"],
    )
