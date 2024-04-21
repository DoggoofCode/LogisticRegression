import numpy as np


def sigmoid(value: float) -> float:
    return 1 / (1 + np.exp(-value))


def fitting(**kwargs):
    global THETA
    for index in range(kwargs["EPOCHS"]):
        z = np.dot(kwargs["x"], THETA)
        h = sigmoid(z)
        gradient = np.dot(kwargs['x'], (h - kwargs['y'][index])) / 3
        THETA -= gradient * kwargs["lr"]


def main(**kwargs):
    x_training = kwargs['x']
    y_training = kwargs['y']
    fitting(x=x_training, y=y_training, EPOCHS=kwargs['epochs'], lr=kwargs["lr"])


if __name__ == "__main__":
    global THETA
    X_train = np.array([
        [0.2, 0.3, 0.1],
        [0.1, 0.2, 0.4],
        [0.4, 0.5, 0.2],
        [0.5, 0.3, 0.1],
    ])

    Y_train = np.array([0, 1, 0, 1])

    LR = 0.01
    EPOCHS = 1000
    m, n = X_train.shape
    THETA = np.zeros(n)

    main(x=X_train, y=Y_train, lr=LR, epochs=EPOCHS)
