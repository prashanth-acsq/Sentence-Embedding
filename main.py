import os
import sys
import numpy as np

from typing import Union
from sentence_transformers import SentenceTransformer
from gensim.parsing.preprocessing import remove_stopwords


INPUT_PATH: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "input")
MODEL_PATH: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models/all-MiniLM-L6-v2")
MODEL = SentenceTransformer(MODEL_PATH)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float: return np.dot(a, b.reshape(-1, 1)) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(query: Union[str, list]):
    if isinstance(query, list):
        query = [kb_item.lower() for kb_item in query]
        query = [remove_stopwords(kb_item) for kb_item in query]
    else:
        query = query.lower()
        query = remove_stopwords(query)
    return MODEL.encode(query)


def main():
    
    args_1: tuple = ("--query1", "-q1")
    args_2: tuple = ("--query2", "-q2")
    args_3: tuple = ("--get-embedding", "-ge")

    query_1: str = "This is sentence 1"
    query_2: str = "This is sentence 2"
    do_get_embedding: bool = False

    if args_1[0] in sys.argv: query_1 = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: query_1 = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: query_2 = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: query_2 = sys.argv[sys.argv.index(args_2[1]) + 1]

    if not do_get_embedding:
        if query_1[-4:] == ".txt":
            assert query_1 in os.listdir(INPUT_PATH)
            with open(os.path.join(INPUT_PATH, query_1), "r") as f: query_1 = f.read()

        if query_2[-4:] == ".txt":
            assert query_2 in os.listdir(INPUT_PATH)
            with open(os.path.join(INPUT_PATH, query_2), "r") as f: query_2 = f.read()
        
        print("\n" + 50*"*" + "\n")
        print(f"Similarity : {cosine_similarity(get_embedding(query_1), get_embedding(query_2))[0]:.5f}")
        print("\n" + 50*"*" + "\n")
    else:
        np.save(get_embedding(query_1), f"{query_1[:10]}_Embedding.npy")
    

if __name__ == "__main__":
    sys.exit(main() or 0)
