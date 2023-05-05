from sentence_transformers import SentenceTransformer, util

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

model = SentenceTransformer("./msmarco-distilbert-cos-v5")


def get_similarities(query, embeddings):
    query_emb = model.encode(query)

    scores = util.dot_score(query_emb, embeddings)[0].cpu().tolist()

    results = list(zip(docs, scores))
    return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    embeddings = model.encode(docs)
    print(embeddings.shape)
    print('------------')
    for query in queries:
        print(query)
        results = get_similarities(query, embeddings)
        for doc, score in results:
            print(score, doc)
        print('------------')
