import numpy as np
def split_test_train(X, y, test_ratio, random_state):
    '''
    Split the data into test and train sets given a test ratio and a random seed
    Returns X_train, y_train, X_test, y_test in that order
    '''
    # setting a random seed to make the shuffling deterministic
    np.random.seed(random_state)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    # shuffling indices so the test-train split is random
    np.random.shuffle(indices)

    test_size = int(num_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test, (num_samples - test_size), test_size