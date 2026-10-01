import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def evaluate_model(
    model,
    preprocessor,
    X,
    y,
    n_splits,
    scoring,
    random_state,
    n_jobs,
):
    """
    Оценивает модель с помощью Stratified K-Fold cross-validation.

    Returns
    -------
    mean_score : float
        Среднее значение метрики по всем fold.
    std_score : float
        Стандартное отклонение метрики.
    scores : np.ndarray
        Значения метрики для каждого fold.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    return scores.mean(), scores.std(), scores