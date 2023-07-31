from argparse import ArgumentParser
import datetime
import pickle
from rank_bm25 import BM25Okapi
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import top_k_accuracy_score
import numpy as np

nltk.download("punkt")

PICKLE_LOCATION = "data/pickles"


def build_claims(pmids: list, claims_corpus: list) -> list:
    """
    Builds a corpus from a list of claims
    """
    print("Prepping claims")
    tokenized_corpus = [
        (pmid, word_tokenize(claim)) for pmid, claim in zip(pmids, claims_corpus)
    ]
    return tokenized_corpus


def build_abstracts(pmids: list, abstracts: list) -> list:
    """
    Builds a corpus from a list of claims
    """
    print("Prepping abstracts")
    tokenized_corpus = [
        (pmid, word_tokenize(abstract)) for pmid, abstract in zip(pmids, abstracts)
    ]
    return tokenized_corpus


def build_corpus(
    pmid_frame: list, claim_frame: list, abstract_frame: list
) -> tuple[list, list]:
    """
    Builds a corpus from a list of claims and abstracts
    """
    print("Prepping corpus")
    return build_claims(pmid_frame, claim_frame), build_abstracts(
        pmid_frame, abstract_frame
    )


def example_bm25():
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?",
    ]

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    query = "windy London"
    tokenized_query = query.split(" ")

    print(f"Scores for query {bm25.get_scores(tokenized_query)}")
    print(f"Most relevant document {bm25.get_top_n(tokenized_query, corpus, n=1)}")


def import_all_data():
    """
    Imports all data from the csv files
    """
    print("Importing claims")
    all_claims = pd.read_csv("data/generated_claims.csv")
    all_claims = all_claims.drop("Unnamed: 0", axis=1)

    print("Importing abstracts")
    all_abstracts = pd.read_csv("data/abstracts-for-retrieval.csv")
    all_abstracts = all_abstracts.drop_duplicates(subset="pmid", keep="first")
    print("Data imported")

    print("Joining data")
    df = pd.merge(all_claims, all_abstracts, on="pmid", how="left")
    df = df.dropna()
    print("Data joined")

    print("Preview data for model")
    print(df.head(20))
    print(df.shape)
    print("Data ready")
    return df


def build_model(abstracts: list):
    """
    Builds a bm25 model from a list of abstracts
    """
    print("Building model")
    abstracts = [abstract for _, abstract in abstracts]
    bm25 = BM25Okapi(abstracts)
    print("Model built")
    return bm25


def query_model(model, query, abstracts):
    """
    Queries a bm25 model with a query
    """
    print("Querying model")
    print(f"Query: {' '.join(query)}")
    print(f"Scores for query {model.get_scores(query)}")
    abstracts = [abstract for _, abstract in abstracts]
    top_abstract = model.get_top_n(query, abstracts, n=1).pop()
    print(f"Most relevant document {' '.join(top_abstract)}")


def get_abstracts_for_query(model, query, abstracts, n=1):
    """
    Queries a bm25 model with a query
    """
    print("Running inference for top abstracts")
    abstracts = [abstract for _, abstract in abstracts]
    return model.get_top_n(query, abstracts, n=n).pop()


def predict_abstract_scores_for_query(model, query, abstracts) -> list:
    """
    Queries a bm25 model with a query
    """
    print("Running inference for scores")
    abstracts = [abstract for _, abstract in abstracts]
    return model.get_scores(query)


def build_model_from_file():
    print("Loading model")
    with open(f"{PICKLE_LOCATION}/bm25_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded")
    return model


def build_abstracts_from_file():
    print("Loading abstracts")
    with open(f"{PICKLE_LOCATION}/tokenized_abstracts.pkl", "rb") as f:
        abs = pickle.load(f)
    print("Abstracts loaded")
    return abs


def build_test_claims_from_file():
    print("Loading unseen claims")
    with open(f"{PICKLE_LOCATION}/test_claims.pkl", "rb") as f:
        claims = pickle.load(f)
    print("Claims loaded")
    return claims


def create_splits(target_data: list, name: str, size: float = 0.8) -> None:
    """
    Creates train and test splits from a list of data
    """
    print("Creating splits")
    # create splits
    train = target_data[: int(len(target_data) * size)]
    test = target_data[int(len(target_data) * size) :]
    # save splits
    with open(f"{PICKLE_LOCATION}/train_{name}.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(f"{PICKLE_LOCATION}/test_{name}.pkl", "wb") as f:
        pickle.dump(test, f)
    print("Splits created")


def get_abstract_for_pmid(abstracts: list[tuple], pmid: str):
    """
    Gets an abstract for a given pmid
    """
    for abstract in abstracts:
        if abstract[0] == pmid:
            return abstract[1]


def get_abstract_index(abstracts: list[tuple], pmid: str) -> int:
    """
    Gets an abstract for a given pmid
    """
    # get row index for pmid
    for i, abstract in enumerate(abstracts):
        if abstract[0] == pmid:
            return i

def get_top_k_abstracts(abstract_scores: list, k: int) -> list:    
    return (-abstract_scores).argsort()[:k]

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Evidence Retrieval Experiments",
        description="Run experiments for evidence retrieval",
        epilog="Example: python bm25_experiment --mode train",
    )
    parser.add_argument("--mode", type=str, default="predict", required=True)
    parser.add_argument("--k_precision", type=int, default=1, required=False)
    args = parser.parse_args()

    print("Start time: ")
    print(datetime.datetime.now())
    if args.mode == "generate-splits":
        df = import_all_data()
        claims, abstracts = build_corpus(
            df["pmid"].to_list(),
            df["generated_claim"].to_list(),
            df["abstract"].to_list(),
        )
        create_splits(claims, "claims")
    elif args.mode == "train":
        df = import_all_data()
        claims, abstracts = build_corpus(
            df["pmid"].to_list(),
            df["generated_claim"].to_list(),
            df["abstract"].to_list(),
        )
        model = build_model(abstracts)

        print("Storing large datasets to file")
        # save model as pickle file for later use
        with open(f"{PICKLE_LOCATION}/bm25_model.pkl", "wb") as f:
            pickle.dump(model, f)
        # save abstracts as pickle file for later use
        with open(f"{PICKLE_LOCATION}/tokenized_abstracts.pkl", "wb") as f:
            pickle.dump(abstracts, f)

    elif args.mode == "query":
        model = build_model_from_file()
        abstracts = build_abstracts_from_file()
        query = word_tokenize(
            "I read a lot about earlier this year that there seems to be indicators of that doctors cant inform their conditions before these months to arrive"
        )
        query_model(model, query=query, abstracts=abstracts)

    elif args.mode == "inference":
        model = build_model_from_file()
        abstracts = build_abstracts_from_file()
        claims = build_test_claims_from_file()
        for pmid, claim in claims[:20]:
            abstract = get_abstracts_for_query(model, query=claim, abstracts=abstracts)
            actual_abstract = get_abstract_for_pmid(abstracts, pmid)
            scores = predict_abstract_scores_for_query(
                model, query=claim, abstracts=abstracts
            )
            # get index of highest number in scores
            index = np.argmax(scores)
            expected_abstract_index = get_abstract_index(abstracts, pmid)
            if actual_abstract == abstract:
                print(f"Claim: {' '.join(claim)}")
                print(f"Actual abstract: {' '.join(actual_abstract)}")
                print(f"Predicted abstract: {' '.join(abstract)}")
                print(f"Index: {index}")
                print(f"Index: {expected_abstract_index}")
                print("Match found")

    elif args.mode == "evaluate":
        k = args.k_precision
        model = build_model_from_file()
        abstracts = build_abstracts_from_file()
        claims = build_test_claims_from_file()
        found = 0
        current_made_predicitions = 0
        matches = pd.DataFrame(columns=["claim", "pmid"])
        for pmid, claim in claims:
            scores = predict_abstract_scores_for_query(
                model, query=claim, abstracts=abstracts
            )
            expected_abstract_index = get_abstract_index(abstracts, pmid)            
            top_scoring_indexes = get_top_k_abstracts(scores, k=k)
            actual_abstract = get_abstract_for_pmid(abstracts, pmid)            
            if expected_abstract_index in top_scoring_indexes:
                expected_abstract_index = get_abstract_index(abstracts, pmid)
                print(f"Claim: {' '.join(claim)}")
                print(f"Index: {expected_abstract_index}")
                print(f"Actual abstract: {' '.join(actual_abstract)}")
                print("Match found")
                # add to matches dataframe using iloc
                matches.loc[len(matches)] = [claim, pmid]                
                found += 1
            current_made_predicitions += 1

            # every 20th claim write the results to file
            if found % 20 == 0 and found > 0:            
                result = found / current_made_predicitions
                print(f"Current result score: {result}")
                print(f"Predictions made so far: {current_made_predicitions}")

        result = found / len(claims)
        print(f"Final result score: {result}")
        matches.to_csv(f"matches-k-{k}.csv", index=False)
    print("End time: ")
    print(datetime.datetime.now())
