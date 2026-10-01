from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


NUM_COLS = [
    "Age",
    "Fare",
    "Pclass",
    "Family_Size",
    "Cabin_known",
    "Cabin_count",
]

OHE_COLS = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "Age_Bin",
    "Fare_Bin",
]

ORDINAL_COLS = [
    "Family_Size_Grouped",
]

EMBEDDING_NUM_COLS = NUM_COLS.copy()

EMBEDDING_CAT_COLS = OHE_COLS + ORDINAL_COLS


def build_preprocessor():
    """
    Создаёт preprocessing pipeline для числовых,
    категориальных и порядковых признаков.
    """

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ohe_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            ),
        ),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=[
                    ["Alone", "Small", "Medium", "Large"]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUM_COLS),
        ("ohe", ohe_pipeline, OHE_COLS),
        ("ordinal", ordinal_pipeline, ORDINAL_COLS),
    ])

    return preprocessor



def build_embedding_preprocessors():
    """
    Создаёт отдельные preprocessors для числовых
    и категориальных признаков DNN с Embedding.
    """

    numeric_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    return numeric_preprocessor, categorical_preprocessor