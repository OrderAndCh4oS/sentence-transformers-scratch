# HuggingFace SentenceTransformers Scratch

This is a short script for experimentation with HuggingFace SentenceTransformers. The purpose of this script is to test out the capabilities of these transformers, specifically measuring and comparing the semantic similarity between sentences.

## Overview

The script uses a pre-trained `msmarco-distilbert-cos-v5` model from the `sentence_transformers` library to encode sentences into embeddings. These embeddings are then used to determine similarities between sets of sentences.

## Usage

1. Define two lists: one containing documents (`docs`) and another containing queries (`queries`). These can be any sentences you would like to compare.
2. Execute the script, which will compute and display results in the form of similarity scores between each query and all documents in `docs`. Higher scores indicate higher semantic similarity.

## Sample Code

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("./msmarco-distilbert-cos-v5")

def get_similarities(query, embeddings):
    query_emb = model.encode(query)
    scores = util.dot_score(query_emb, embeddings)[0].cpu().tolist()
    results = list(zip(docs, scores))
    return sorted(results, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    docs = [
        "The cat climbed onto the roof, gracefully.",
        "She eagerly browsed through her favourite bookshop.",
        "Torrential rain caused severe flooding in the village.",
    ]

    queries = [
        "Effortlessly, the feline ascended to the rooftop.",
        "In her beloved bookshop, she perused with enthusiasm.",
        "The downpour led to substantial inundation within the hamlet.",
    ]

    embeddings = model.encode(docs)
    
    print(embeddings.shape)
    
    print('------------')
    
     for query in queries:
         print(query)
         results = get_similarities(query, embeddings)
         for doc, score in results:
             print(score, doc)
         print('------------')
```

## Output

The script will display the similarity scores of each query compared to all documents. The higher the score, the more semantically similar the sentences are.
