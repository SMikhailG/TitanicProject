import pandas as pd


def load_data(train_path, test_path):
    """
    Загружает train и test датасеты.

    Parameters
    ----------
    train_path : Path
        Путь к train.csv.
    test_path : Path
        Путь к test.csv.

    Returns
    -------
    train_df : pd.DataFrame
        Обучающий датасет.
    test_df : pd.DataFrame
        Тестовый датасет.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def split_features_target(train_df, test_df):
    """
    Отделяет целевую переменную и удаляет признаки,
    которые не используются для обучения моделей.
    """

    drop_cols = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
        "SibSp",
        "Parch",
    ]

    X = train_df.drop(
        columns=["Survived"] + drop_cols
    )

    y = train_df["Survived"]

    X_test = test_df.drop(
        columns=drop_cols
    )

    return X, y, X_test