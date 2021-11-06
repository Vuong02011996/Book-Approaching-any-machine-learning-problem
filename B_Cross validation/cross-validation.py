import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn import datasets


def create_k_folds():
    # csv with image id, image location and image label.
    df = pd.read_csv("train.csv")

    # create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "kfold"] = fold

    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)


def create_stratified_k_folds():
    # csv with image id, image location and image label.
    df = pd.read_csv("train.csv")

    # create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y = df.target.values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, "kfold"] = fold

    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)


def create_stratified_k_fold_for_regression(data):
    # create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Sturge's rule
    # I take the floor of the value, you can also just round it.
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin tagets
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "kfold"] = f

    # drop the bins column
    data = data.drop("bins", axis=1)

    # return data frame with folds
    return data


if __name__ == '__main__':
    # We create a sample dataset with 1500 samples and 100 features and 1 target
    X, y = datasets.make_regression(n_samples=1500, n_features=100, n_targets=1)

    # create a data frame out of our numpy arrays
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

    df.loc[:, "target"] = y

    # create folds
    df = create_stratified_k_fold_for_regression(data=df)
    a = 0



