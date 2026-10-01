from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "gender_submission.csv"

RESULTS_DIR = OUTPUT_DIR / "results"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"


# General settings
RANDOM_STATE = 123

N_SPLITS = 5
SCORING = "accuracy"


# Feature engineering
AGE_BINS = 8
FARE_BINS = 6


# Models
MODELS = [
    "logistic_regression",
    "knn",
    "catboost",
    "lightgbm",
    "xgboost",
]


# CatBoost
CATBOOST_PARAMS = {
    "depth": 4,
    "iterations": 300,
    "l2_leaf_reg": 7,
    "learning_rate": 0.05,
}


# LightGBM
LIGHTGBM_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 4,
    "min_child_samples": 5,
    "n_estimators": 200,
    "num_leaves": 7,
}


# XGBoost
XGBOOST_PARAMS = {
    "colsample_bytree": 1.0,
    "learning_rate": 0.1,
    "max_depth": 2,
    "n_estimators": 200,
    "subsample": 0.8,
}

CV_N_JOBS = -1

# DNN
DNN_CONFIG = {
    "hidden_size1": 32,
    "hidden_size2": 16,
    "activation": "leaky_relu",
    "dropout_rate": 0.3,
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "bce_with_logits",
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
}

# DNN with Embedding
EMBEDDING_DNN_CONFIG = {
    "hidden_size1": 32,
    "hidden_size2": 16,
    "activation": "leaky_relu",
    "dropout_rate": 0.3,
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "bce_with_logits",
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "max_embedding_dim": 50,
}