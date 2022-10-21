import autograd.numpy as np

def mse_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    # assert that we have an array of at least scalars
    assert y_true.ndim >= 1
    return np.sum(np.mean(np.square(y_pred - y_true), axis=-1))


def test_mse_loss():
    y_true1 = np.array([[0., 1.], [0., 0.]])
    y_pred1 = np.array([[1., 1.], [1., 0.]])

    y_true2 = np.array([[0., 10.], [0., 0.]])
    y_pred2 = np.array([[10., 10.], [10., 0.]])
    mse = mse_loss

    print(
        f"mse between y_true1 and y_pred2  with own: {mse(y_true1, y_pred1)}")

    print(
        f"mse between y_true1 and y_pred2  with own: {mse(y_true2, y_pred2)}")


if __name__ == "__main__":
    test_mse_loss()