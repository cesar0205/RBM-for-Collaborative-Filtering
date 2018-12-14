import tensorflow as tf
import numpy as np
from scipy.sparse import load_npz
from misc import batch_iterator

'''
    Computes precision, recall and f1 score for the rbm recommender system. 
'''

def calculate_classification_metrics(A_test_visible, A_test_hidden):

    restore_saver = tf.train.import_meta_graph("./rbm_cl/final_model/final_rbm_cl_model.meta")

    X_in = tf.get_default_graph().get_tensor_by_name("X_in:0")
    prediction = tf.get_default_graph().get_tensor_by_name("prediction/Tensordot:0")

    with tf.Session() as session:
        restore_saver.restore(session, "./rbm_cl/final_model/final_rbm_cl_model")

        tot_measures = np.zeros(3)
        no_batches = 0;
        shortlist = 50;

        for X_batch_test, X_batch_valid in batch_iterator(A_test_visible, A_test_hidden, batch_sz=256):

            predictions = session.run(
                prediction,
                feed_dict={X_in: X_batch_test}
            )

            #Calculate predicted positives. Threshold > 3
            inds_rec = np.where((predictions > 3) & (X_batch_valid > 0))[:shortlist]
            ind_rec_ = [(i, j) for i, j in zip(inds_rec[0], inds_rec[1])]

            #Calculate ground truth positives
            ind_like = np.where(X_batch_valid > 3)
            ind_like_ = [(i, j) for i, j in zip(ind_like[0], ind_like[1])]

            #Calculate ground truth negatives
            ind_dislike = np.where((X_batch_valid <= 3) & (X_batch_valid > 0))
            ind_dislike_ = [(i, j) for i, j in zip(ind_dislike[0], ind_dislike[1])]

            #Support
            cnt = len(ind_like_) + len(ind_dislike_)

            #Calculate True positives
            tp = set(ind_rec_).intersection(set(ind_like_))
            tp_ = len(tp)
            #Calculate False positives
            fp = set(ind_rec_).intersection(set(ind_dislike_))
            fp_ = len(fp)
            #Calculate False negatives
            fn = set(ind_like_) ^ (set(ind_rec_).intersection(set(ind_like_)))
            fn_ = len(fn)

            precision = 0;
            if (tp_ + fp_ > 0):
                precision = float(tp_) / (tp_ + fp_)

            recall = 0;
            if (tp_ + fn_):
                recall = float(tp_) / (tp_ + fn_)

            f1 = 0
            if (recall + precision > 0):
                f1 = 2.0 * precision * recall / (precision + recall)

            tot_measures += np.array([precision, recall, f1])
            no_batches += 1;
            if (no_batches % 10 == 0):
                print("Processed...", no_batches)

    tot_measures /= no_batches;
    return tot_measures;

def main():
    A_test_visible = load_npz("./sparse_datasets/A_test_visible.npz")
    A_test_hidden = load_npz("./sparse_datasets/A_test_hidden.npz")
    metrics = calculate_classification_metrics(A_test_visible, A_test_hidden)
    print("Precision: {:.4f} \nRecall: {:.4f} \nF1: {:.4f}".format(metrics[0], metrics[1], metrics[2]))

if __name__ == "__main__":
    main();