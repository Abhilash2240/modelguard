from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor(
    numerical_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer for tabular preprocessing.

    Numerical features:
    - Median imputation for missing values
    - Standard scaling (mean=0, variance=1)

    Categorical features:
    - Constant imputation ('missing')
    - OneHotEncoder with handle_unknown='ignore' and dense array output
    """
    num_cols = numerical_cols or NUMERICAL_FEATURES
    cat_cols = categorical_cols or CATEGORICAL_FEATURES

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor
