import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

def main():

    print("Reading movie ratings file...")
    # https://www.kaggle.com/grouplens/movielens-20m-dataset
    df = pd.read_csv('./movielens/ratings_large.csv')

    # Preprocess the data.

    # 1. Make movies id go from 0 - n_movies - 1
    # 2. Make users id go from 0 - n_users - 1
    print("Creating movie indexes...")
    movie2idx = {}

    # Map from the movie id to its index
    movie_ids = set(df.movieId.values)
    for idx, movie_id in enumerate(movie_ids):
        movie2idx[movie_id] = idx
    print("Applying movie2idx...")
    df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

    df.userId = df.userId - 1

    N = df.userId.max() + 1  # number of users
    M = df.movie_idx.max() + 1  # number of movies

    print("Train test split...")
    df_train, df_test = train_test_split(df, test_size=0.2)
    n_train = len(df_train)
    n_test = len(df_test)

    print("Creating sparse matrices..")
    # Use a lil-matrix. As stated in the documentation this is a structure for constructing sparse matrices incrementally.
    A_train = lil_matrix((N, M))
    A_test = lil_matrix((N, M))


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
    df_train.apply(assign_ratings(A_train, n_train), axis=1)
    print("Transforming test data")
    df_test.apply(assign_ratings(A_test, n_test), axis=1)

    # Finally we convert it to CSR format format for fast arithmetic and matrix vector operations
    A_train = A_train.tocsr()
    A_test = A_test.tocsr()
    save_npz("A_train.npz", A_train)
    save_npz("A_test.npz", A_test)

if __name__ == '__main__':
    main()