from kedro.pipeline import node, Pipeline
from titanic_with_kedro.nodes import preprocess


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess.preprocess,
                inputs="train",
                outputs="train_prep",
                name="preprocess",
            ),
        ],
        tags=['de_tag'],
    )
