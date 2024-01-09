import argparse
import json
import os

from datasets import load_dataset
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy.sparse import vstack
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--text_column_name", default="text", type=str)
    parser.add_argument("--output_folder", type=str, default="outputs")

    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = f"{args.output_folder}/classical"
    os.makedirs(output_folder, exist_ok=True)

    np.random.seed(41)
    metrics = {}

    # load dataset, tokenizer, model
    dataset = load_dataset(args.dataset_name)

    # rename text column to `text`
    if args.text_column_name != "text":
        dataset = dataset.rename_column(args.text_column_name, "text")

    train_df = dataset["train"].to_pandas()
    valid_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    features = {"BoW": CountVectorizer, "TFIDF": TfidfVectorizer}

    for feature_name, feature in zip(features.keys(), features.values()):
        vectorizer = feature(tokenizer=word_tokenize, token_pattern=None)
        vectorizer.fit(train_df.text)

        X_train = vectorizer.transform(train_df.text)
        X_valid = vectorizer.transform(valid_df.text)
        X_test = vectorizer.transform(test_df.text)

        y_train, y_valid, y_test = train_df.label, valid_df.label, test_df.label

        # follows https://github.com/IndoNLP/nusa-writes/blob/main/boomer/main_boomer.py#L114C5-L122C20
        classifiers = {
            "nb": MultinomialNB(),
            "svm": SVC(),
            "lr": LogisticRegression(),
        }

        param_grids = {
            "nb": {"alpha": np.linspace(0.001, 1, 50)},
            "svm": {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["rbf", "linear"]},
            "lr": {"C": [0.01, 0.1, 1, 10, 100]},
        }

        for c in classifiers:
            X = vstack([X_train, X_valid])
            y = list(y_train) + list(y_valid)

            ps = PredefinedSplit([-1] * len(y_train) + [0] * len(y_valid))
            clf = GridSearchCV(classifiers[c], param_grids[c], cv=ps, n_jobs=-1)
            clf.fit(X, y)

            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary")
            metrics[f"{c}_{feature_name}_accuracy"] = accuracy
            metrics[f"{c}_{feature_name}_f1"] = f1
            metrics[f"{c}_{feature_name}_precision"] = precision
            metrics[f"{c}_{feature_name}_recall"] = recall

    with open(f"{output_folder}/eval_results_{args.dataset_name.split('/')[-1]}.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
