from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def get_model(
    model_name,
    random_state,
    catboost_params=None,
    lightgbm_params=None,
    xgboost_params=None,
):
    """
    Создаёт модель по её имени.
    """

    if model_name == "logistic_regression":
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
        )

    if model_name == "knn":
        return KNeighborsClassifier()

    if model_name == "catboost":
        return CatBoostClassifier(
            random_state=random_state,
            verbose=False,
            **catboost_params,
        )

    if model_name == "lightgbm":
        return LGBMClassifier(
            random_state=random_state,
            verbose=-1,
            **lightgbm_params,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
            **xgboost_params,
        )

    raise ValueError(f"Unknown model: {model_name}")