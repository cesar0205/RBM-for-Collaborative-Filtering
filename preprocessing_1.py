import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

def main():

    '''
        Creates a sparse matrix of N (Users) x M (Movies) and saves it to disk.
    '''


    print("Reading movie ratings file...")
    # https://www.kaggle.com/grouplens/movielens-20m-dataset
    df = pd.read_csv('./movie_lens/ratings_large.csv')

    # Preprocess the data.

    # 1. Make movies id go from 0 - n_movies - 1
    # 2. Make users id go from 0 - n_users - 1
    print("Creating movie indexes...")
    movie2idx = {}

    # Map from the movie id to its index
    movie_ids = set(df.movieId.values)

    for idx, movie_id in enumerate(movie_ids):
        movie2idx[movie_id] = idx

    df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

    df.userId = df.userId - 1

    N = df.userId.max() + 1  # number of users
    M = df.movie_idx.max() + 1  # number of movies

    n_total = len(df)

    print("Creating sparse matrix..")
    # Use a lil-matrix. As stated in the documentation this is a structure for constructing sparse matrices incrementally.
    A_original = lil_matrix((N, M))

    def assign_ratings(A, total):
        assign_ratings.i = 0;

        def process_data(row):
            assign_ratings.i += 1;
            if assign_ratings.i % 100000 == 0:
                part = float(assign_ratings.i) / total
                print("Processed: {:.3f}%".format(100 * part))
            i = int(row.userId)
            j = int(row.movie_idx)
            A[i, j] = row.rating

        return process_data

    print("Transforming training data")
    df.apply(assign_ratings(A_original, n_total), axis=1)

    # Finally we convert it to CSR format format for fast arithmetic and matrix vector operations
    A_train = A_original.tocsr()
    save_npz("A_orignal.npz", A_train)

if __name__ == '__main__':
    main()