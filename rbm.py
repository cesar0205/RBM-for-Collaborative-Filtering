import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from datetime import datetime

'''
RBM for Collaborative filtering implementation.  
Takes the A_train and A_test sparse matrices created with the preprocessing.py file and creates an RBM model to
predict future movie ratings.
For CF methods MSE is applicable as well other metrics as precision, recall and f2 measure.
In particular we are going to use MSE.

At the end of each epoch we evaluate the model and check if it is the best current model. It is possible to do it at 
batch level (every some interval of batches). Although the latter approach can find a better global model, 
it is really computationally expensive and not worth the effort. 
'''

# N Number of samples
# D Number of movies --- Visible units
# K Number of Categories
# M Number of Hidden units

# W Weight matrix (D, K, M)
# c (M) Bias matrix for hidden units
# b (D, K) Bias matrix for visible units


def batch_iterator(X, X_test, batch_sz):
    N = X.shape[0]
    for i in range(0, N, batch_sz):
        start_ind = i;
        end_ind = min(i + batch_sz, N);
        yield X[start_ind:end_ind].toarray(), X_test[start_ind:end_ind].toarray();

class RBM(object):
    '''
    RBM for Collaborative filtering implementation.
    '''
    def __init__(self, D, M, K):
        self.D = D  # input feature size
        self.M = M  # hidden size
        self.K = K  # number of ratings

        self.checkpoint_path = "./rbm_cl/checkpoints/rbm_cl_model.ckpt";
        self.final_model_path = "./rbm_cl/final_model/final_rbm_cl_model";
        self.saved_losses_path = "./rbm_cl/saved_losses_file.epoch";

    def _dot1(V, W):
        # V is N x D x K (batch of visible units)
        # W is D x K x M (weights)
        # returns N x M (hidden layer size)
        return tf.tensordot(V, W, axes=[[1, 2], [0, 1]])

    def _dot2(H, W):
        # H is N x M (batch of hiddens)
        # W is D x K x M (weights transposed)
        # returns N x D x K (visible)
        return tf.tensordot(H, W, axes=[[1], [2]])

    def build(self):
        # params
        D = self.D  # input feature size
        M = self.M  # hidden size
        K = self.K  # number of ratings

        self.W = tf.Variable(tf.random_normal(shape=(D, K, M)) * np.sqrt(2.0 / M), name="W")
        self.c = tf.Variable(np.zeros(M).astype(np.float32), name="c")
        self.b = tf.Variable(np.zeros((D, K)).astype(np.float32), name="b")
        learning_rate = 1e-2;

        # Train and test data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D), name="X_in")
        self.X_test = tf.placeholder(tf.float32, shape=(None, D), name="X_test")

        with tf.name_scope("positive_phase"):
            # ====== 1. Clamp visible units
            # One ot encode X
            X = tf.cast(self.X_in * 2 - 1, tf.int32)
            V = tf.one_hot(X, K, name="V")

            # =======2. Calculate p(h == 1| v) --> N x M
            positive_hidden = tf.nn.sigmoid(self._dot1(V, self.W) + self.c, name="positive_hidden")

            # =======3. Draw a sample from p(h == 1| v)
            # Remember each movie (column) is independent to each other
            # We need to create a vector of probabilities on each movie, such that the sum of the probablities of the posible
            # ratings to a movie is one.
            r = tf.random_uniform(shape=tf.shape(positive_hidden))
            hidden_states = tf.to_float(positive_hidden > r, name="hidden_states")

        with tf.name_scope("negative_phase"):
            # =======4. Calculate p(v == 1 | h) --> N x D x K
            reconstruction_logits = tf.add(self._dot2(hidden_states, self.W), self.b, name="reconstruction_logits")

            # =======5. Draw a sample from p(v == 1| h)
            # We don't have to actually do the softmax of the logits as the Catgorical function takes care of it..
            cdist = tf.distributions.Categorical(logits=reconstruction_logits)
            reconstruction = cdist.sample()  # shape is (N, D)
            reconstruction = tf.one_hot(reconstruction, depth=K)  # turn it into (N, D, K)
            # As the reconstruction also gave ratings to missing ratings we should mask them so they don't contribute to objective

            mask2d = tf.cast(self.X_in > 0, tf.float32)
            mask3d = tf.stack([mask2d] * K, axis=-1)  # repeat K times in last dimension

            reconstruction = tf.multiply(reconstruction, mask3d,
                                         name="reconstruction")  # missing ratings shouldn't contribute to objective

        with tf.name_scope("objective"):
            # The objective is equal to -log(p(v)) which we try to minimize
            positive_phase_free_energy = tf.reduce_mean(self._free_energy(V), name="pfe");
            negative_phase_free_energy = tf.reduce_mean(self._free_energy(reconstruction), name="nfe");

            self.objective = tf.subtract(positive_phase_free_energy, negative_phase_free_energy, name="objective");
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.objective, name="train_op")

        # We will monitor the model using the xentropy loss as the objective function changes sign every batch so it is difficult
        # to observe if the model is making progress.
        # We can't use the mse to monitor de model as it is only calculated at the end of the epoch and we need to monitor the
        # system preferably each batch.
        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=V, logits=reconstruction_logits, name="entropy")
            self.loss = tf.reduce_mean(xentropy, name="loss")

        with tf.name_scope("prediction"):
            reconstruction_probs = tf.nn.softmax(reconstruction_logits)
            self.one_to_ten = tf.constant((np.arange(10) + 1).astype(np.float32) / 2)
            # (N, D, K) .dot (K, ) ---> (N, D, )
            # At this point prediction are not integers numbers but float.
            self.prediction = tf.tensordot(reconstruction_probs, self.one_to_ten, axes=[[2], [0]])

        with tf.name_scope("sse"):
            # Train SSE
            mask = tf.cast(self.X_in > 0, tf.float32)
            se = mask * (self.X_in - self.prediction) * (self.X_in - self.prediction)
            self.sse = tf.reduce_sum(se, name="sse")

            # test SSE
            mask = tf.cast(self.X_test > 0, tf.float32)
            tse = mask * (self.X_test - self.prediction) * (self.X_test - self.prediction)
            self.test_sse = tf.reduce_sum(tse, name="test_sse")

        with tf.name_scope("save_and_init"):
            self.initop = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.saver = saver;

    def fit(self, X, X_test, epochs=10, batch_sz=256, print_interval=20, show_fig=True):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build();

        self.session = tf.Session(graph=self.graph);

        train_mses = [];
        test_mses = [];

        start_epoch = 0;
        best_mse = np.infty;

        if os.path.isfile(self.saved_losses_path):
            # if the checkpoint file exists, restore the model and load counters
            print("Statistics file ", self.saved_losses_path, " exists. Restoring model..")
            state = self._restore_model_and_statistics();

            train_mses = state["train_mses"];
            test_mses = state["test_mses"];

            start_epoch = len(train_mses);
            best_mse = state["best_mse"];

            print("Training was interrupted. Continuing at epoch", start_epoch)

        else:
            print("Statistics file ", self.checkpoint_path, " doesn't exist. Starting again..")
            self.session.run(self.initop);

        N, D = X.shape
        n_batches = int(np.ceil(N / batch_sz))

        for epoch in range(start_epoch, epochs):
            t0 = datetime.now()
            print("Computing epoch:", epoch)
            X, X_test = shuffle(X, X_test)  # everything has to be shuffled

            # Reset counters
            sse = 0
            test_sse = 0

            n = 0
            test_n = 0

            curr_batch = 0;
            for X_batch, X_test_batch in batch_iterator(X, X_test, batch_sz):

                _, loss, = self.session.run(
                    [self.train_op, self.loss],
                    feed_dict={self.X_in: X_batch, self.X_test: X_test_batch}
                )

                # We need to run sse and test_sse in a different run because otherwise we would use
                # the old "reconstruction_logits" used to calculate the "loss" value,
                # and of course we want a updated value of "reconstruction_logits" (after the train_op is run)
                # to calculate the new sse and test_sse.

                batch_sse, batch_tsse = self.session.run(
                    [self.sse, self.test_sse],
                    feed_dict={self.X_in: X_batch, self.X_test: X_test_batch}
                )

                # number of train ratings
                n += np.count_nonzero(X_batch);
                test_n += np.count_nonzero(X_test_batch)

                sse += batch_sse
                test_sse += batch_tsse

                if (curr_batch % print_interval == 0):
                    print("Batches {:d}/{:d}, Loss:{:.6f} ".format(curr_batch, n_batches, loss))
                curr_batch += 1

            train_mse = sse / n;
            test_mse = test_sse / test_n;

            train_mses.append(train_mse)
            test_mses.append(test_mse)

            if test_mse < best_mse:
                print("Saving best model, mse {:.6f} ".format(test_mse))
                self._save_best_model();
                best_mse = test_mse;

            self._save_model(epoch, best_mse, train_mse, test_mse);

            print("Finished epoch:", epoch, " Train mse:", train_mse, " Test mse:", test_mse, " Duration",
                  datetime.now() - t0)

            print()

        self._restore_best_model();

        if show_fig:
            plt.plot(train_mses, label='train mse')
            plt.plot(test_mses, label='test mse')
            plt.legend()
            plt.show()

    def _restore_best_model(self):
        self.saver.restore(self.session, self.final_model_path)

    def _restore_model_and_statistics(self):
        state = {}
        self.saver.restore(self.session, self.checkpoint_path)

        train_mses = []
        test_mses = []
        if os.path.isfile(self.saved_losses_path):
            with open(self.saved_losses_path, "rb") as f:
                statistics = f.readlines()

                for statistic in statistics:
                    elements = statistic.split()
                    epoch = int(elements[0])
                    best_mse = float(elements[1])
                    train_mse = float(elements[2])
                    test_mse = float(elements[3])
                    train_mses.append(train_mse)
                    test_mses.append(test_mse)

                state["train_mses"] = train_mses;
                state["test_mses"] = test_mses;
                state["epoch"] = epoch;
                state["best_mse"] = best_mse;
                print(state)

        else:
            state["train_costs"] = [];
            state["test_costs"] = [];

        return state;

    def _save_model(self, epoch, best_mse, train_mse, test_mse):
        self.saver.save(self.session, self.checkpoint_path)

        with open(self.saved_losses_path, "ab+") as f:
            f.write(b"%d %.6f %.6f %.6f\n" % (epoch, best_mse, train_mse, test_mse))

    def _save_best_model(self):
        self.saver.save(self.session, self.final_model_path)

    def _free_energy(self, V):
        # V is N x D x K
        # b is D, K
        # W is D, K, M
        first_term = -tf.reduce_sum(self._dot1(V, self.b))
        second_term = -tf.reduce_sum(
            tf.nn.softplus(self._dot1(V, self.W) + self.c),
            axis=1
        )
        return first_term + second_term

    def predict(self, X):
        prediction = self.session.run(self.prediction, feed_dict={self.X_in: X})
        return np.round(prediction).astype(np.int)


def main():
    A_train = load_npz("A_train.npz")
    A_test = load_npz("A_test.npz")
    print("Done loading data...");
    N, M = A_train.shape
    rbm = RBM(M, 50, 10)
    rbm.fit(A_train, A_test)

if __name__ == '__main__':
    main()