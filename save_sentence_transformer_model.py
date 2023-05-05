"""
Some models:

MiniLM-L6-H384-uncased
MiniLM-L12-H384-uncased
msmarco-distilbert-cos-v5
msmarco-MiniLM-L6-cos-v5
msmarco-MiniLM-L12-cos-v5
multi-qa-MiniLM-L6-cos-v1
"""

from sentence_transformers import SentenceTransformer
modelName = "msmarco-distilbert-cos-v5"
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
model.save(modelName)
# model = SentenceTransformer(modelName)
