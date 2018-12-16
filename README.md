# RMB-for-Collaborative-Filtering
A tensorflow-implementation of the RBM for Collaborative Filtering

This RBM implementation was inspired by the paper Restricted Boltzmann machines for collaborative filtering by Salakhutdinov, R., Mnih, A., and Hinton for the Netflix competition. It was one of the front runners in 2006.

It is trained with the Kaggle movielens-20m-dataset https://www.kaggle.com/grouplens/movielens-20m-dataset. This is a huge dataset to work with, so the model is trained using batches of 256 samples.

For proper training and evaluation. The data is divided in training data (60%), evaluation data (20%) and test data (20%)

There are two preprocessing steps:
1) The data is converted to a sparse matrix.
2) The sparce matrix is divided in three parts, training, evaluation and test.
   For the evaluation and test sets, half of the ratings are hidden and used as target ratings.

The model exposes two main methods: fit and predict.

The error curves are shown next:

![MSE error](mse_error.png)


For final evaluation, precision, recalll and f1 score were calculated using the test set. A rating greater than 3 is consider positive.

Precision: 0.7106\
Recall: 0.9178\
F1: 0.8010



