from kedro.pipeline import node, Pipeline
from titanic_with_kedro.nodes import preprocess
from titanic_with_kedro.nodes import modeling


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess.feature_target_split,
                inputs="train",
                outputs=["features_train", "y_train"],
                name="feature_target_split",
            ),
            node(
                func=preprocess.take_initial_character,
                inputs="features_train",
                outputs="train_trans",
                name="take_initial_character_train",
            ),
            node(
                func=preprocess.split_numerical_categorical,
                inputs="train_trans",
                outputs=["train_num", "train_cat"],
                name="split_num_cat_train",
            ),
            node(
                func=preprocess.preprocess_numerical_train,
                inputs="train_num",
                outputs=["train_prepped_num", "preprocessor_numerical"],
                name="prep_numerical_train",
            ),
            node(
                func=preprocess.preprocess_categorical_train,
                inputs="train_cat",
                outputs=["train_prepped_cat", "preprocessor_categorical"],
                name="prep_categorical_train",
            ),
            node(
                func=preprocess.concat_prepped,
                inputs=["train_prepped_num", "train_prepped_cat"],
                outputs="train_prepped",
                name="concat_prepped_train",
            ),
            node(
                func=modeling.model_construction,
                inputs=["train_prepped", "y_train", "parameters"],
                outputs="classifier",
                name="model_construction",
            ),
        ],
        tags=['modeling_tag'],
    )
