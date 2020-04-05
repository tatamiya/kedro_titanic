from kedro.pipeline import node, Pipeline
from titanic_with_kedro.nodes import preprocess
from titanic_with_kedro.nodes import modeling


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess.take_initial_character,
                inputs="test",
                outputs="test_trans",
                name="take_initial_character_test",
            ),
            node(
                func=preprocess.split_numerical_categorical,
                inputs="test_trans",
                outputs=["test_num", "test_cat"],
                name="split_num_cat_test",
            ),
            node(
                func=preprocess.preprocess_numerical_test,
                inputs=["test_num", "preprocessor_numerical"],
                outputs="test_prepped_num",
                name="prep_numerical_test",
            ),
            node(
                func=preprocess.preprocess_categorical_test,
                inputs=["test_cat", "preprocessor_categorical"],
                outputs="test_prepped_cat",
                name="prep_categorical_test",
            ),
            node(
                func=preprocess.concat_prepped,
                inputs=["test_prepped_num", "test_prepped_cat"],
                outputs="test_prepped",
                name="concat_prepped_test",
            ),
            node(
                func=modeling.prediction,
                inputs=["test_prepped", "classifier", "test"],
                outputs="prediction_result",
                name="prediction",
            ),
        ],
        tags=['prediction_tag'],
    )
