from scipy.sparse import csr_matrix, save_npz, load_npz
import numpy as np
from sklearn.utils import shuffle

def main():

    '''
        Divides the data into train, test and validation sets.
    '''

    def split_ratings(A):
        index_test = np.where(A > 0)
        n_ratings = len(index_test[0])
        perm = np.random.permutation(n_ratings)
        x_ind = index_test[0][perm]
        y_ind = index_test[1][perm]
        n_split = n_ratings // 2
        xi_visible = x_ind[:n_split]
        yi_visible = y_ind[:n_split]
        xi_hidden = x_ind[n_split:]
        yi_hidden = y_ind[n_split:]
        A_hidden = np.copy(A)
        A[xi_hidden, yi_hidden] = 0
        A_hidden[xi_visible, yi_visible] = 0
        return A, A_hidden

    A_original = load_npz("A_original.npz")

    A_shuffled = shuffle(A_original)
    N, M = A_shuffled.shape

    n_test = int(N * 0.2)

    A_val = A_shuffled[-2 * n_test: -n_test].toarray()
    A_test = A_shuffled[-n_test:].toarray()
    A_train = A_shuffled[:-2 * n_test]

    A_test_visible, A_test_hidden = split_ratings(A_test)
    A_val_visible, A_val_hidden = split_ratings(A_val)
    save_npz("./sparse_datasets/A_test_visible.npz", csr_matrix(A_test_visible))
    save_npz("./sparse_datasets/A_test_hidden.npz", csr_matrix(A_test_hidden))
    save_npz("./sparse_datasets/A_val_visible.npz", csr_matrix(A_val_visible))
    save_npz("./sparse_datasets/A_val_hidden.npz", csr_matrix(A_val_hidden))
    save_npz("./sparse_datasets/A_train.npz", A_train)

if __name__ == '__main__':
    main()