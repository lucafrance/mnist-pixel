import os
import pickle

import mnist_reader

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor


class MnistData():
    
    def __init__(self, normalize=True):
        # db_name = "mnist" or "fashion"
        # kind = "train" or "t10k"
        self.db = dict()
        for db_name in ["mnist", "fashion"]:
            for kind in ["train", "t10k"]:
                print("Loading {} database, kind {}...".format(db_name, kind))
                data_path = os.path.join("data", db_name)
                X, y = mnist_reader.load_mnist(data_path, kind=kind)
                if normalize:
                    X = preprocessing.normalize(X, norm="l2")
                self.db[(db_name, kind)] = (X, y)
    
    
    def __call__(self, db_name, kind):
        return self.db[(db_name, kind)]


class MonofeatureLogisticRegression():
    """Class to try logistic regression on just one feature at the time 
    and pick the best one (i.e. the biggest in sample accuracy)"""
    
    def __init__(self):
        self.best_clf = None
        self.best_score = -1
        self.best_feature_idx = -1
    
    
    def fit(self, X, y):
        num_features = X.shape[1]
        X_clms = [X[:,i].reshape(-1, 1) for i in range(num_features)]
        
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_features):
                futures.append(executor.submit(LogisticRegression(solver="sag").fit, X_clms[i], y))
    
            for i in range(num_features):
                clf = futures[i].result()
                score = clf.score(X_clms[i], y)
                if score > self.best_score:
                    self.best_clf = clf
                    self.best_score = score
                    self.best_feature_idx = i
    
                if (i+1) % 100 == 0 or (i+1) == num_features:
                    print("Checked {} / {} features".format(i+1, num_features))
    
        return self     
    
            
    def score(self, X, y):
        X_clm = X[:,self.best_feature_idx].reshape(-1, 1)
        return self.best_clf.score(X_clm, y)


def evaluate_logistic_all(db_name, train_data, test_data):
    
    results = []

    X_train, y_train = train_data
    X_test, y_test = test_data
    
    pickle_name = "pickles/logistic_standard_all_{}.pickle".format(db_name)
    if os.path.exists(pickle_name):
        print("Loading {}...".format(pickle_name))
        clf = pickle.load(open(pickle_name, "rb"))
    else:
        print("Training standard logistic for all data of {}...".format(db_name))
        clf = LogisticRegression(solver="sag", n_jobs=-1).fit(X_train, y_train)
        print("Dumping {}...".format(pickle_name))
        pickle.dump(clf, open(pickle_name, "wb"))
    
    print("Evaluating standard logistic for all data of {}...".format(db_name))
    in_sample_acc = clf.score(X_train, y_train)
    out_sample_acc = clf.score(X_test, y_test)
    
    results.append("Database {}, standard logistic regression, all data, in sample accuracy: {:.4f}".format(db_name, in_sample_acc))
    results.append("Database {}, standard logistic regression, all data, out of sample accuracy: {:.4f}".format(db_name, out_sample_acc))
    
    return results


def evaluate_logistic_1px_pairwise(db_name, train_data, test_data):
    
    results = []
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    pickle_name = "pickles/logistic_1px_pairwise_{}.pickle".format(db_name)
    if os.path.exists(pickle_name):
        print("Loading {}...".format(pickle_name))
        clfs = pickle.load(open(pickle_name, "rb"))
    else:
        clfs = [[None for _ in range(10)] for _ in range(10)]
        for i in range(10):
            for j in range(i+1, 10):
                print("Training 1-pixel logistic pairwise ({},{}) for {}...".format(i, j, db_name))
                relevant_rows_idxs = np.logical_or(np.equal(y_train, i), np.equal(y_train, j)) 
                X_pair_train = X_train[relevant_rows_idxs]
                y_pair_train = y_train[relevant_rows_idxs]
                clfs[i][j] = MonofeatureLogisticRegression().fit(X_pair_train, y_pair_train)
        print("Dumping {}...".format(pickle_name))
        pickle.dump(clfs, open(pickle_name, "wb"))
    
    in_sample_accs = np.full((10, 10), np.nan)
    out_sample_accs = np.full((10, 10), np.nan)
    for i in range(10):
        for j in range(i+1, 10):
            print("Evaluating 1-pixel logistic pairwise ({},{}) for {}...".format(i, j, db_name))
            relevant_rows_idxs = np.logical_or(np.equal(y_train, i), np.equal(y_train, j)) 
            X_pair_train = X_train[relevant_rows_idxs]
            y_pair_train = y_train[relevant_rows_idxs]
            relevant_rows_idxs = np.logical_or(np.equal(y_test, i), np.equal(y_test, j)) 
            X_pair_test = X_test[relevant_rows_idxs]
            y_pair_test = y_test[relevant_rows_idxs]
            in_sample_acc = clfs[i][j].score(X_pair_train, y_pair_train)
            out_sample_acc = clfs[i][j].score(X_pair_test, y_pair_test)
            in_sample_accs[i, j] = in_sample_acc
            out_sample_accs[i, j] = out_sample_acc
            in_sample_accs[j, i] = in_sample_accs[i, j]
            out_sample_accs[j, i] = out_sample_accs[i, j]
    
    csv_name_in_sample = "csv/accuracy_1px_pairwise_{}_in.csv".format(db_name)
    csv_name_out_sample = "csv/accuracy_1px_pairwise_{}_out.csv".format(db_name)
    np.savetxt(csv_name_in_sample, in_sample_accs, fmt="%.4f")
    np.savetxt(csv_name_out_sample, out_sample_accs, fmt="%.4f")
    results.append("Database {}, 1-pixel logistic regression pairwise")
    results.append(" - accuracy saved to files \"{}\" and \"{}\"".format(db_name, csv_name_in_sample, csv_name_out_sample))
    results.append(" - in sample: min {:.4f}, max {:.4f}, mean {:.4f}, median {:.4f}".format(np.nanmin(in_sample_accs), np.nanmax(in_sample_accs), 
                                                                                             np.nanmean(in_sample_accs), np.nanmedian(in_sample_accs)))
    results.append(" - out of sample: min {:.4f}, max {:.4f}, mean {:.4f}, median {:.4f}".format(np.nanmin(out_sample_accs), np.nanmax(out_sample_accs), 
                                                                                                 np.nanmean(out_sample_accs), np.nanmedian(out_sample_accs)))
    
    return results
    
def evaluate_logistic_1px_all(db_name, train_data, test_data):
    results = []
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    pickle_name = "pickles/logistic_1px_all_{}.pickle".format(db_name)
    
    if os.path.exists(pickle_name):
        print("Loading {}...".format(pickle_name))
        clf = pickle.load(open(pickle_name, "rb"))
    else:
        print("Training 1-pixel logistic for all data of {}...".format(db_name))
        clf = MonofeatureLogisticRegression().fit(X_train, y_train)
        print("Dumping {}...".format(pickle_name))
        pickle.dump(clf, open(pickle_name, "wb"))
        
    
    print("Evaluating 1-pixel logistic for all data of {}...".format(db_name))
    in_sample_acc = clf.score(X_train, y_train)
    out_sample_acc = clf.score(X_test, y_test)
    
    results.append("Database {}, 1-pixel logistic regression, in sample accuracy: {:.4f}".format(db_name, in_sample_acc))
    results.append("Database {}, 1-pixel logistic regression, out of sample accuracy: {:.4f}".format(db_name, out_sample_acc))
    
    return results

if __name__ == "__main__":
    
    os.makedirs("csv", exist_ok=True)
    os.makedirs("pickles", exist_ok=True)
    
    results = []
    
    data = MnistData()
    for db_name in ["mnist", "fashion"]:
        train_data = data(db_name, "train")
        test_data = data(db_name, "t10k")
        results += evaluate_logistic_all(db_name, train_data, test_data) 
        results += evaluate_logistic_1px_pairwise(db_name, train_data, test_data)
        results += evaluate_logistic_1px_all(db_name, train_data, test_data)
    open("results.txt", "wt").write("\n".join(results))

