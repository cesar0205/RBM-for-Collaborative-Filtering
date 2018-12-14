def batch_iterator(X = None, X_test = None, batch_sz = 256):
    N = X.shape[0]
    for i in range(0, N, batch_sz):
        start_ind = i;
        end_ind = min(i + batch_sz, N);
        if(X_test is None):
            yield X[start_ind:end_ind].toarray();
        else:
            yield X[start_ind:end_ind].toarray(), X_test[start_ind:end_ind].toarray();