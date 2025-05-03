import pandas as pd

# Load the dataset
df = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Showing first few rows
print(df.head())
print(df.columns)

from sentence_transformers import SentenceTransformer, util

# Loading a small, fast model (great for semantic similarity)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all the sentences
embeddings1 = model.encode(df['text1'].tolist(), convert_to_tensor=True)
embeddings2 = model.encode(df['text2'].tolist(), convert_to_tensor=True)

import torch

# Computing cosine similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Diagonal values are what we need (similarity of each pair)
similarity_scores = cosine_scores.diag().cpu().numpy()

# Adding to the dataframe
df['similarity_score'] = similarity_scores

# Showing final result
print(df[['text1', 'text2', 'similarity_score']].head())

# Save similarity scores for analysis or debugging
df.to_csv("similarity_scored_output.csv", index=False)

#  Saving the model to reuse later in API
model.save("semantic_model/")
