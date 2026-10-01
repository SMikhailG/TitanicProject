import numpy as np
import pandas as pd


def create_family_features(df):
    """
    Создаёт признаки, связанные с размером семьи пассажира.
    """
    df = df.copy()

    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

    def family_group(size):
        if size == 1:
            return "Alone"
        elif size <= 4:
            return "Small"
        elif size <= 6:
            return "Medium"
        else:
            return "Large"

    df["Family_Size_Grouped"] = df["Family_Size"].apply(family_group)

    return df


def create_age_features(train_df, test_df, n_bins):
    """
    Заполняет пропуски Age медианой из train
    и создаёт возрастные интервалы.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    median_age = train_df["Age"].median()

    train_df["Age"] = train_df["Age"].fillna(median_age)
    test_df["Age"] = test_df["Age"].fillna(median_age)

    _, bins = pd.qcut(
        train_df["Age"],
        q=n_bins,
        retbins=True,
        duplicates="drop",
    )

    bins[[0, -1]] = [-np.inf, np.inf]

    train_df["Age_Bin"] = pd.cut(
        train_df["Age"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    test_df["Age_Bin"] = pd.cut(
        test_df["Age"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    train_df["Age_Bin"] = train_df["Age_Bin"].astype("object")
    test_df["Age_Bin"] = test_df["Age_Bin"].astype("object")

    return train_df, test_df


def create_fare_features(train_df, test_df, n_bins):
    """
    Заполняет пропуски Fare и создаёт интервалы стоимости билета.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    median_fare_by_class = train_df.groupby("Pclass")["Fare"].median()

    test_df["Fare"] = test_df["Fare"].fillna(
        test_df["Pclass"].map(median_fare_by_class)
    )

    _, bins = pd.qcut(
        train_df["Fare"],
        q=n_bins,
        retbins=True,
        duplicates="drop",
    )

    bins[[0, -1]] = [-np.inf, np.inf]

    train_df["Fare_Bin"] = pd.cut(
        train_df["Fare"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    test_df["Fare_Bin"] = pd.cut(
        test_df["Fare"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    train_df["Fare_Bin"] = train_df["Fare_Bin"].astype("object")
    test_df["Fare_Bin"] = test_df["Fare_Bin"].astype("object")

    return train_df, test_df


def create_title_features(df):
    """
    Извлекает титул пассажира из имени
    и объединяет редкие титулы.
    """
    df = df.copy()

    df["Title"] = (
        df["Name"]
        .str.split(",", expand=True)[1]
        .str.split(".", expand=True)[0]
        .str.strip()
    )

    title_map = {
        "Capt": "Military",
        "Col": "Military",
        "Major": "Military",
        "Jonkheer": "Noble",
        "the Countess": "Noble",
        "Don": "Noble",
        "Lady": "Noble",
        "Sir": "Noble",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    df["Title"] = df["Title"].replace(title_map)

    return df


def create_cabin_features(df):
    """
    Создаёт признаки на основе информации о каюте.
    """
    df = df.copy()

    df["Cabin_known"] = df["Cabin"].notna().astype(int)

    df["Cabin_count"] = df["Cabin"].fillna("").apply(
        lambda x: len(x.split()) if x != "" else 0
    )

    df["Cabin"] = df["Cabin"].fillna("U")
    df["Deck"] = df["Cabin"].str[0]

    return df


def build_features(train_df, test_df, age_bins, fare_bins):
    """
    Выполняет весь feature engineering для train и test.
    """
    train_df = create_family_features(train_df)
    test_df = create_family_features(test_df)

    train_df, test_df = create_age_features(
        train_df,
        test_df,
        n_bins=age_bins,
    )

    train_df, test_df = create_fare_features(
        train_df,
        test_df,
        n_bins=fare_bins,
    )

    train_df = create_title_features(train_df)
    test_df = create_title_features(test_df)

    train_df = create_cabin_features(train_df)
    test_df = create_cabin_features(test_df)

    return train_df, test_df