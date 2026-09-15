import numpy as np


traindata = np.loadtxt("traindata.txt")

X = traindata[:, :8]
Y = traindata[:, 8]


def least_squares(Z, y):
    w, _, _, _ = np.linalg.lstsq(
        Z,
        y,
        rcond=None
    )
    return w


def calculate_mse(w, Z, y):
    predictions = Z @ w
    return np.mean(
        (predictions - y) ** 2
    )


def final_49_basis(X):

    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x3 = X[:, 2:3]
    x4 = X[:, 3:4]
    x5 = X[:, 4:5]
    x6 = X[:, 5:6]
    x7 = X[:, 6:7]
    x8 = X[:, 7:8]

    return np.hstack([

        np.ones((len(X), 1)),

        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x8,

        x1 ** 2,
        x2 ** 2,
        x3 ** 2,
        x6 ** 2,
        x8 ** 2,

        np.log1p(np.abs(x1)),
        np.log1p(np.abs(x2)),
        np.log1p(np.abs(x3)),
        np.log1p(np.abs(x4)),
        np.log1p(np.abs(x7)),
        np.log1p(np.abs(x8)),

        np.sqrt(np.abs(x2)),
        np.sqrt(np.abs(x3)),
        np.sqrt(np.abs(x4)),
        np.sqrt(np.abs(x6)),
        np.sqrt(np.abs(x7)),
        np.sqrt(np.abs(x8)),

        np.sin(x1),
        np.sin(x3),
        np.sin(x6),
        np.sin(x8),

        np.cos(x4),
        np.cos(x5),
        np.cos(x6),
        np.cos(x7),
        np.cos(x8),

        1.0 / (1.0 + np.abs(x1)),
        1.0 / (1.0 + np.abs(x2)),
        1.0 / (1.0 + np.abs(x4)),
        1.0 / (1.0 + np.abs(x5)),
        1.0 / (1.0 + np.abs(x6)),
        1.0 / (1.0 + np.abs(x7)),
        1.0 / (1.0 + np.abs(x8)),

        x1 * x7,
        x2 * x8,
        x1 * x4,
        x3 * x4,
        x2 * x5,
        x2 * x7,
        x4 * x8,
        x3 * x8
    ])


K = 5
RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)

indices = np.arange(
    len(X)
)

rng.shuffle(
    indices
)

folds = np.array_split(
    indices,
    K
)

cv_mse_values = []


for k in range(K):

    val_idx = folds[k]

    train_idx = np.concatenate([
        folds[j]
        for j in range(K)
        if j != k
    ])

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]

    Z_train = final_49_basis(
        X_train
    )

    Z_val = final_49_basis(
        X_val
    )

    w = least_squares(
        Z_train,
        Y_train
    )

    val_mse = calculate_mse(
        w,
        Z_val,
        Y_val
    )

    cv_mse_values.append(
        val_mse
    )


cross_validated_mse = np.mean(
    cv_mse_values
)


Z_all = final_49_basis(
    X
)

w_final = least_squares(
    Z_all,
    Y
)

training_mse = calculate_mse(
    w_final,
    Z_all,
    Y
)


print(
    "Training MSE:",
    training_mse
)

print(
    "Cross-validated test MSE:",
    cross_validated_mse
)


X_test = np.loadtxt(
    "testinputs.txt"
)

Z_test = final_49_basis(
    X_test
)

Y_predictions = (
    Z_test @ w_final
)

np.savetxt(
    "predictions.txt",
    Y_predictions,
    fmt="%.10f"
)

print(
    "Saved",
    len(Y_predictions),
    "predictions to predictions.txt"
)