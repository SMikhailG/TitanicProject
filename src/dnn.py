import copy
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from src.utils import set_seed
from src.preprocessing import EMBEDDING_CAT_COLS, EMBEDDING_NUM_COLS, build_embedding_preprocessors


def get_activation(name):
    """
    Возвращает функцию активации по имени.
    """
    if name == "leaky_relu":
        return nn.LeakyReLU()

    if name == "relu":
        return nn.ReLU()

    raise ValueError(f"Unknown activation: {name}")


def get_loss(name):
    """
    Возвращает функцию потерь по имени.
    """
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unknown loss: {name}")


def get_optimizer(name, model, learning_rate):
    """
    Создаёт optimizer по имени.
    """
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def get_scheduler(name, optimizer, factor, patience):
    """
    Создаёт scheduler по имени.
    """
    if name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
        )

    raise ValueError(f"Unknown scheduler: {name}")


class TitanicDNN(nn.Module):
    """
    Полносвязная нейронная сеть для задачи бинарной классификации Titanic.
    """

    def __init__(
         self,
        input_size,
        hidden_size1,
        hidden_size2,
        dropout_rate,
        activation,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size1, hidden_size2),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size2, 1),
        )

    def forward(self, x):
        return self.network(x)


class TitanicEmbeddingDNN(nn.Module):

    """
    Нейронная сеть с embedding-слоями
    для категориальных признаков Titanic.
    """

    def __init__(
        self,
        num_numeric_features,
        category_sizes,
        embedding_dims,
        hidden_size1,
        hidden_size2,
        dropout_rate,
        activation,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=category_size,
                embedding_dim=embedding_dim,
            )
            for category_size, embedding_dim in zip(
                category_sizes,
                embedding_dims,
            )
        ])

        total_embedding_dim = sum(embedding_dims)

        input_size = (
            num_numeric_features
            + total_embedding_dim
        )

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size1, hidden_size2),
            get_activation(activation),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size2, 1),
        )

    def forward(self, x_numeric, x_categorical):

        embedded_features = []

        for i, embedding in enumerate(self.embeddings):
            embedded = embedding(x_categorical[:, i])
            embedded_features.append(embedded)

        x = torch.cat(
            [x_numeric] + embedded_features,
            dim=1,
        )

        return self.network(x)


def evaluate_dnn_cv(
    X,
    y,
    preprocessor,
    config,
    n_splits,
    random_state,
    device,
):
    """
    Оценивает обычную DNN с помощью Stratified K-Fold cross-validation.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        set_seed(random_state + fold)

        print(f"\nDNN Fold {fold}/{n_splits}")

        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        fold_preprocessor = copy.deepcopy(preprocessor)

        X_train_processed = fold_preprocessor.fit_transform(
            X_fold_train
        )

        X_valid_processed = fold_preprocessor.transform(
            X_fold_valid
        )

        X_train_tensor = torch.tensor(
            X_train_processed,
            dtype=torch.float32,
        )

        X_valid_tensor = torch.tensor(
            X_valid_processed,
            dtype=torch.float32,
        )

        y_train_tensor = torch.tensor(
            y_fold_train.values,
            dtype=torch.float32,
        ).unsqueeze(1)

        y_valid_tensor = torch.tensor(
            y_fold_valid.values,
            dtype=torch.float32,
        ).unsqueeze(1)

        train_dataset = TensorDataset(
            X_train_tensor,
            y_train_tensor,
        )

        valid_dataset = TensorDataset(
            X_valid_tensor,
            y_valid_tensor,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
        )

        model = TitanicDNN(
            input_size=X_train_tensor.shape[1],
            hidden_size1=config["hidden_size1"],
            hidden_size2=config["hidden_size2"],
            dropout_rate=config["dropout_rate"],
            activation=config["activation"],
        ).to(device)

        loss_fn = get_loss(config["loss"])

        optimizer = get_optimizer(
            name=config["optimizer"],
            model=model,
            learning_rate=config["learning_rate"],
        )

        scheduler = get_scheduler(
            name=config["scheduler"],
            optimizer=optimizer,
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
        )

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(config["epochs"]):

            model.train()

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                logits = model(X_batch)

                loss = loss_fn(
                    logits,
                    y_batch,
                )

                loss.backward()
                optimizer.step()

            model.eval()

            val_loss = 0.0

            with torch.no_grad():

                for X_batch, y_batch in valid_loader:

                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    logits = model(X_batch)

                    loss = loss_fn(
                        logits,
                        y_batch,
                    )

                    val_loss += (
                        loss.item() * X_batch.size(0)
                    )

            val_loss /= len(valid_dataset)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(
                    model.state_dict()
                )

        model.load_state_dict(best_model_state)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for X_batch, y_batch in valid_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities >= 0.5
                ).float()

                correct += (
                    predictions == y_batch
                ).sum().item()

                total += y_batch.size(0)

        fold_accuracy = correct / total

        scores.append(fold_accuracy)

        print(
            f"DNN Fold {fold} Accuracy: "
            f"{fold_accuracy:.4f}"
        )

        del model
        del optimizer
        del scheduler
        del best_model_state

        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = np.array(scores)

    return scores.mean(), scores.std(), scores



def evaluate_embedding_dnn_cv(
    X,
    y,
    config,
    n_splits,
    random_state,
    device,
):
    """
    Оценивает DNN с Embedding с помощью Stratified K-Fold cross-validation.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        set_seed(random_state + fold)

        print(f"\nEmbedding DNN Fold {fold}/{n_splits}")

        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        numeric_preprocessor, categorical_preprocessor = (
            build_embedding_preprocessors()
        )

        # Числовые признаки
        X_train_numeric = numeric_preprocessor.fit_transform(
            X_fold_train[EMBEDDING_NUM_COLS]
        )

        X_valid_numeric = numeric_preprocessor.transform(
            X_fold_valid[EMBEDDING_NUM_COLS]
        )

        # Категориальные признаки
        X_train_categorical = categorical_preprocessor.fit_transform(
            X_fold_train[EMBEDDING_CAT_COLS]
        )

        X_valid_categorical = categorical_preprocessor.transform(
            X_fold_valid[EMBEDDING_CAT_COLS]
        )

        # -1 означает неизвестную категорию.
        # Сдвигаем все индексы на +1, чтобы 0 был отдельным индексом unknown.
        X_train_categorical = X_train_categorical.astype(int) + 1
        X_valid_categorical = X_valid_categorical.astype(int) + 1

        # Для каждого категориального признака узнаём количество категорий,
        # которые OrdinalEncoder встретил на обучающей части fold.
        # +1 добавляем для специального индекса 0,
        # который используем для неизвестных категорий в validation/test.
        category_sizes = [
            len(categories) + 1
            for categories in categorical_preprocessor
                .named_steps["encoder"]
                .categories_
        ]


        # Для каждого категориального признака выбираем размер embedding-вектора.
        # Используем примерно половину от количества категорий,
        # но ограничиваем размер сверху значением max_embedding_dim из config.
        embedding_dims = [
            min(
                config["max_embedding_dim"],
                (size + 1) // 2,
            )
            for size in category_sizes
        ]

        X_train_numeric_tensor = torch.tensor(
            X_train_numeric,
            dtype=torch.float32,
        )

        X_valid_numeric_tensor = torch.tensor(
            X_valid_numeric,
            dtype=torch.float32,
        )

        X_train_categorical_tensor = torch.tensor(
            X_train_categorical,
            dtype=torch.long,
        )

        X_valid_categorical_tensor = torch.tensor(
            X_valid_categorical,
            dtype=torch.long,
        )

        y_train_tensor = torch.tensor(
            y_fold_train.values,
            dtype=torch.float32,
        ).unsqueeze(1)

        y_valid_tensor = torch.tensor(
            y_fold_valid.values,
            dtype=torch.float32,
        ).unsqueeze(1)

        train_dataset = TensorDataset(
            X_train_numeric_tensor,
            X_train_categorical_tensor,
            y_train_tensor,
        )

        valid_dataset = TensorDataset(
            X_valid_numeric_tensor,
            X_valid_categorical_tensor,
            y_valid_tensor,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
        )

        model = TitanicEmbeddingDNN(
            num_numeric_features=len(EMBEDDING_NUM_COLS),
            category_sizes=category_sizes,
            embedding_dims=embedding_dims,
            hidden_size1=config["hidden_size1"],
            hidden_size2=config["hidden_size2"],
            dropout_rate=config["dropout_rate"],
            activation=config["activation"],
        ).to(device)

        loss_fn = get_loss(
            config["loss"]
        )

        optimizer = get_optimizer(
            name=config["optimizer"],
            model=model,
            learning_rate=config["learning_rate"],
        )

        scheduler = get_scheduler(
            name=config["scheduler"],
            optimizer=optimizer,
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
        )

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(config["epochs"]):

            model.train()

            for (
                X_numeric_batch,
                X_categorical_batch,
                y_batch,
            ) in train_loader:

                X_numeric_batch = X_numeric_batch.to(device)
                X_categorical_batch = X_categorical_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                logits = model(
                    X_numeric_batch,
                    X_categorical_batch,
                )

                loss = loss_fn(logits, y_batch)

                loss.backward()
                optimizer.step()

            model.eval()

            val_loss = 0.0

            with torch.no_grad():

                for (
                    X_numeric_batch,
                    X_categorical_batch,
                    y_batch,
                ) in valid_loader:

                    X_numeric_batch = X_numeric_batch.to(device)
                    X_categorical_batch = X_categorical_batch.to(device)
                    y_batch = y_batch.to(device)

                    logits = model(
                        X_numeric_batch,
                        X_categorical_batch,
                    )

                    loss = loss_fn(logits, y_batch)

                    val_loss += (
                        loss.item()
                        * X_numeric_batch.size(0)
                    )

            val_loss /= len(valid_dataset)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_model_state = copy.deepcopy(
                    model.state_dict()
                )

        model.load_state_dict(best_model_state)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for (
                X_numeric_batch,
                X_categorical_batch,
                y_batch,
            ) in valid_loader:

                X_numeric_batch = X_numeric_batch.to(device)
                X_categorical_batch = X_categorical_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(
                    X_numeric_batch,
                    X_categorical_batch,
                )

                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities >= 0.5
                ).float()

                correct += (
                    predictions == y_batch
                ).sum().item()

                total += y_batch.size(0)

        fold_accuracy = correct / total

        scores.append(fold_accuracy)

        print(
            f"Embedding DNN Fold {fold} Accuracy: "
            f"{fold_accuracy:.4f}"
        )

        del model
        del optimizer
        del scheduler
        del best_model_state

        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = np.array(scores)

    return scores.mean(), scores.std(), scores