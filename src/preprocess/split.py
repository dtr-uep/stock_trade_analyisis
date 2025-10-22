from sklearn.model_selection import KFold

def purge_kfold_split(data, n_splits, embargo=0.01):
    embargo_size = int(len(data) * embargo)
    count_down = n_splits
    kf = KFold(n_splits=n_splits + 1, shuffle=False)

    for test_idx, train_idx in kf.split(data):

        if count_down == 0:
            return
        test_idx = test_idx[test_idx > (train_idx[-1] + int(embargo_size / 2))]
        train_idx = train_idx[train_idx < (test_idx[0] - int(embargo_size / 2))]

        count_down -= 1

        yield train_idx, test_idx