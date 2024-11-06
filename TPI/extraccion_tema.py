import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Lista de tweets de ejemplo
# tweets = [
#     "Artificial intelligence is transforming industries",
#     "Machine learning drives innovation in the tech sector",
#     "Natural language processing makes AI smarter and more intuitive",
#     "Cybersecurity is crucial in the modern digital landscape",
#     "Quantum computing could revolutionize technology",
#     "Cloud computing allows companies to scale their services easily",
#     "Blockchain technology is changing the way we think about data security",
#     "5G networks are set to improve internet speeds significantly",
#     "The football season is heating up with some great matches",
#     "The Olympics are a celebration of global athletic talent",
#     "Basketball games have been intense this season",
#     "Tennis tournaments bring out the best in each player",
#     "The new soccer stadium will host the championship finals",
#     "Fans are excited for the upcoming World Cup",
#     "Training and discipline are key for any top athlete",
#     "The marathon was a true test of endurance and strength",
#     "The new policy aims to improve public healthcare",
#     "Government officials are debating the new tax reforms",
#     "The presidential election is just around the corner",
#     "Protests continue in the city over economic policies"
# ]
tw = pd.read_csv('Guía8/only_tweets_short.csv', header=None)

# Convertir la columna única a una lista
tweets = tw[0].tolist()  # Accedemos a la primera (y única) columna


# Cargar el tokenizer y el modelo de BERT preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Función para obtener los embeddings de BERT para cada tweet
def get_bert_embeddings(tweets):
    embeddings = []
    it = 0
    for tweet in tweets:
        # Tokenización y creación de tensores
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Obtener las representaciones del modelo
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extraer la representación de la última capa
        last_hidden_state = outputs.last_hidden_state
        
        # Promediar las representaciones de todas las palabras en el tweet
        tweet_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(tweet_embedding)
        print(it)
        it+=1
    return np.array(embeddings)

# Obtener los embeddings de los tweets
tweet_embeddings = get_bert_embeddings(tweets)

# Usar KMeans para clasificar los tweets en temas
num_topics = 3  # Número de temas que quieres identificar
kmeans = KMeans(n_clusters=num_topics, random_state=42)
kmeans.fit(tweet_embeddings)

# Usar TF-IDF para obtener las palabras clave
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(tweets)

# Obtener los términos (palabras) en el vocabulario
terms = vectorizer.get_feature_names_out()

# Para cada cluster, obtener las palabras más relevantes
def get_top_keywords_for_clusters(kmeans, tfidf_matrix, terms, n_words=5):
    cluster_keywords = {}
    
    for i in range(kmeans.n_clusters):
        # Obtener el índice de los tweets que pertenecen a este cluster
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        
        # Obtener el promedio de los vectores TF-IDF de todos los tweets en el cluster
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1  # A1 para convertir a 1D array
        
        # Obtener los índices de las palabras más relevantes (mayor TF-IDF)
        top_indices = cluster_tfidf.argsort()[-n_words:][::-1]
        
        # Obtener las palabras más relevantes
        top_words = [terms[idx] for idx in top_indices]
        cluster_keywords[i] = top_words
    
    return cluster_keywords

# Obtener las palabras clave de cada cluster
cluster_keywords = get_top_keywords_for_clusters(kmeans, tfidf_matrix, terms)

# Mostrar los resultados
for idx, tweet in enumerate(tweets):
    print(f"Tweet: {tweet}")
    print(f"Tema asignado: Tema {kmeans.labels_[idx] + 1}")
    print("-" * 50)

# Mostrar las palabras clave de cada cluster
for cluster_id, keywords in cluster_keywords.items():
    print(f"Cluster {cluster_id + 1} - Palabras clave: {', '.join(keywords)}")
