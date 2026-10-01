from config import (
    AGE_BINS,
    CATBOOST_PARAMS,
    FARE_BINS,
    LIGHTGBM_PARAMS,
    MODELS,
    N_SPLITS,
    RANDOM_STATE,
    SCORING,
    TEST_PATH,
    TRAIN_PATH,
    XGBOOST_PARAMS,
    CV_N_JOBS,
    RESULTS_DIR,
    DNN_CONFIG,
    EMBEDDING_DNN_CONFIG,
    SUBMISSIONS_DIR,
)

import pandas as pd

from src.data import load_data, split_features_target
from src.features import build_features
from src.models import get_model
from src.preprocessing import build_preprocessor
from src.train import evaluate_model
from src.dnn import evaluate_dnn_cv, evaluate_embedding_dnn_cv
from src.utils import get_device, set_seed

from sklearn.pipeline import Pipeline


def main():

    set_seed(RANDOM_STATE)

    device = get_device()

    print(f"Device: {device}")

    train_df, test_df = load_data(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
    )

    train_df, test_df = build_features(
        train_df=train_df,
        test_df=test_df,
        age_bins=AGE_BINS,
        fare_bins=FARE_BINS,
    )

    test_passenger_ids = test_df["PassengerId"].copy()

    X, y, X_test = split_features_target(
        train_df=train_df,
        test_df=test_df,
    )

    preprocessor = build_preprocessor()

    results = []

    for model_name in MODELS:

        model = get_model(
            model_name=model_name,
            random_state=RANDOM_STATE,
            catboost_params=CATBOOST_PARAMS,
            lightgbm_params=LIGHTGBM_PARAMS,
            xgboost_params=XGBOOST_PARAMS,
        )

        mean_score, std_score, scores = evaluate_model(
            model=model,
            preprocessor=preprocessor,
            X=X,
            y=y,
            n_splits=N_SPLITS,
            scoring=SCORING,
            random_state=RANDOM_STATE,
            n_jobs=CV_N_JOBS,
        )

        results.append({
            "Model": model_name,
            "Mean Accuracy": mean_score,
            "STD": std_score,
        })

        print(f"\n{model_name}")
        print("CV scores:", scores)
        print(f"Mean accuracy: {mean_score:.4f}")
        print(f"STD: {std_score:.4f}")


    dnn_mean, dnn_std, dnn_scores = evaluate_dnn_cv(
        X=X,
        y=y,
        preprocessor=preprocessor,
        config=DNN_CONFIG,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        device=device,
    )

    results.append({
        "Model": "dnn",
        "Mean Accuracy": dnn_mean,
        "STD": dnn_std,
    })

    print("\ndnn")
    print("CV scores:", dnn_scores)
    print(f"Mean accuracy: {dnn_mean:.4f}")
    print(f"STD: {dnn_std:.4f}")


    embedding_dnn_mean, embedding_dnn_std, embedding_dnn_scores = (
        evaluate_embedding_dnn_cv(
            X=X,
            y=y,
            config=EMBEDDING_DNN_CONFIG,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            device=device,
        )
    )

    results.append({
        "Model": "dnn_embedding",
        "Mean Accuracy": embedding_dnn_mean,
        "STD": embedding_dnn_std,
    })

    print("\ndnn_embedding")
    print("CV scores:", embedding_dnn_scores)
    print(f"Mean accuracy: {embedding_dnn_mean:.4f}")
    print(f"STD: {embedding_dnn_std:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Mean Accuracy",
        ascending=False,
    )

    # Для финального submission выбираем лучшую классическую модель,
    # так как sklearn Pipeline используется только для моделей из MODELS.
    classic_results = results_df[
        results_df["Model"].isin(MODELS)
    ]

    best_model_name = classic_results.iloc[0]["Model"]

    print(f"\nBest model: {best_model_name}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = RESULTS_DIR / "model_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    best_model = get_model(
        model_name=best_model_name,
        random_state=RANDOM_STATE,
        catboost_params=CATBOOST_PARAMS,
        lightgbm_params=LIGHTGBM_PARAMS,
        xgboost_params=XGBOOST_PARAMS,
    )

    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", best_model),
    ])

    final_pipeline.fit(X, y)

    predictions = final_pipeline.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_passenger_ids,
        "Survived": predictions.astype(int),
    })

    SUBMISSIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission_path = SUBMISSIONS_DIR / "submission.csv"

    submission.to_csv(
        submission_path,
        index=False,
    )

    print(f"Submission saved to: {submission_path}")

    print(f"\nResults saved to: {results_path}")

    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()