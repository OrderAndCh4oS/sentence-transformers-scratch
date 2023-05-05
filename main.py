from sentence_transformers import SentenceTransformer, util

docs = [
    "The cat climbed onto the roof, gracefully.",
    "She eagerly browsed through her favourite bookshop.",
    "Torrential rain caused severe flooding in the village.",
    "Children played joyfully on the sunlit playground.",
    "A vivid rainbow arched across the sky after the storm.",
    "He sipped his tea while reading the morning newspaper.",
    "The artist painted a serene landscape using gentle brushstrokes",
    "The aroma of freshly baked bread filled the kitchen",
    "Elderly couples danced elegantly at the ballroom event",
    "Total darkness enveloped them as they entered the cave",
]

queries = [
    "Effortlessly, the feline ascended to the rooftop.",
    "In her beloved bookshop, she perused with enthusiasm.",
    "The downpour led to substantial inundation within the hamlet.",
    "Youngsters frolicked exuberantly on the sun-drenched play area.",
    "Post-storm, a striking rainbow spanned over the firmament.",
    "Morning headlines were consumed as he savoured his cuppa.",
    "Using delicate strokes, a tranquil vista was crafted by the painter.",
    "Wafting through the room was an irresistible scent of oven-fresh loaves.",
    "Aged pairs waltzed gracefully at the formal dance gathering",
    "Upon entering cavernous depths, inky blackness enclosed their surroundings",
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
