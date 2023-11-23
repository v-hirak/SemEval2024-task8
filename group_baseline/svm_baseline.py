import pandas as pd
import evaluate
import numpy as np
from transformers import set_seed
import os
from sklearn.model_selection import train_test_split
import argparse
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import pickle


def get_data(train_path, test_path, random_seed):
    """
    Function to read dataframe with columns.
    Args:
        train_path (str): Path to training set.
        test_path (str): Path to test set.
        random_seed (int): Random seed number.
    Returns:
        train_df (DataFrame): Training dataframe.
        val_df (DataFrame): Validation dataframe.
        test_df (DataFrame): Test dataframe.
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df


def train(train_df, valid_df, checkpoints_path):
    """
    Function to train SVM baseline model.
    Args:
        train_df (DataFrame): Training set.
        valid_df (DataFrame): Validation set.
        checkpoints_path (str): Checkpoint path to save the model.
    Returns:
        None
    """

    model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
    ('svd', TruncatedSVD(n_components=5000)),
    ('scaler', StandardScaler()),
    ('SVM', LinearSVC(C=1))], verbose=True)

    X_train = np.concatenate((train_df['text'].values, valid_df['text'].values))
    y_train = np.concatenate((train_df['label'].values, valid_df['label'].values))

    model.fit(X_train, y_train)

    # save model
    saved_model_path = checkpoints_path

    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

    with open('{}tfidf_svm_pipeline.pkl'.format(saved_model_path), 'wb') as f:
        pickle.dump(model, f)


def test(test_df, model_path, labels):
    """
    Function to test SVM baseline model.
    Args:
        test_df (DataFrame): Test set.
        model_path (str): Path to saved model.
        labels (List): True labels to evaluate the model.
    Returns:
        results (float): Calculated metric.
        preds (List): Predicted by the model labels.
    """

    with open('{}tfidf_svm_pipeline.pkl'.format(model_path), 'rb') as f:
        pipeline = pickle.load(f)

    preds = pipeline.predict(test_df['text'].values)

    # return dictionary of classification report
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=labels)

    return results, preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)

    args = parser.parse_args()

    random_seed = 0
    train_path =  args.train_file_path  # For example 'subtaskA_train_multilingual.jsonl'
    test_path =  args.test_file_path  # For example 'subtaskA_test_multilingual.jsonl'
    subtask =  args.subtask  # For example 'A'
    prediction_path = args.prediction_file_path  # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(test_path))
        raise ValueError("File doesnt exists: {}".format(test_path))

    set_seed(random_seed)

    # get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    # train detector model
    train(train_df, valid_df, f"Tfidf_SVM_baseline/subtask{subtask}/")

    # test detector model
    results, predictions = test(test_df, f"Tfidf_SVM_baseline/subtask{subtask}/", test_df['label'])

    logging.info(results)

    # save results
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')
