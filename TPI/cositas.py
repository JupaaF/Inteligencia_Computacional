import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel

# Descarga de los recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('stopwords')

tweets = [
    "Artificial intelligence is transforming industries",
    "Machine learning drives innovation in the tech sector",
    "Natural language processing makes AI smarter and more intuitive",
    "Cybersecurity is crucial in the modern digital landscape",
    "Quantum computing could revolutionize technology",
    "Cloud computing allows companies to scale their services easily",
    "Blockchain technology is changing the way we think about data security",
    "5G networks are set to improve internet speeds significantly",
    "The football season is heating up with some great matches",
    "The Olympics are a celebration of global athletic talent",
    "Basketball games have been intense this season",
    "Tennis tournaments bring out the best in each player",
    "The new soccer stadium will host the championship finals",
    "Fans are excited for the upcoming World Cup",
    "Training and discipline are key for any top athlete",
    "The marathon was a true test of endurance and strength",
    "The new policy aims to improve public healthcare",
    "Government officials are debating the new tax reforms",
    "The presidential election is just around the corner",
    "Protests continue in the city over economic policies"
]

# Cargar el archivo CSV y seleccionar los primeros 200 datos
# df = pd.read_csv('TPI/databases/Twitterdatainsheets.csv', usecols=['index', 'TweetID', 'text'], low_memory=False)

# df_200 = df.head(10000)  # Selecciona las primeras 200 filas

df_200 = tweets  # Selecciona las primeras 200 filas

# tweets = df_200['text'].fillna('')  # Rellenar valores faltantes con cadenas vacías

# Obtener las stopwords en español
stop_words = set(stopwords.words('english'))

# Función para preprocesar el texto
def preprocesar_texto(texto):
    tokens = word_tokenize(texto.lower())  # Convierte a minúsculas y tokeniza
    tokens_limpios = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens_limpios

# Preprocesar todos los tweets
tweets_preprocesados = [preprocesar_texto(tweet) for tweet in tweets]

# Crear un diccionario de palabras únicas
diccionario = corpora.Dictionary(tweets_preprocesados)

# Convertir tweets a formato bag-of-words
corpus = [diccionario.doc2bow(tweet) for tweet in tweets_preprocesados]

# Entrenar el modelo LDA
num_topicos = 3  # Cambia el número de tópicos según lo necesites
modelo_lda = LdaModel(corpus=corpus, id2word=diccionario, num_topics=num_topicos, random_state=42, passes=50, iterations=400)


# Mostrar los temas encontrados
for i, topico in modelo_lda.show_topics(formatted=False):
    palabras_clave = [palabra for palabra, probabilidad in topico]
    print(f"Tópico {i + 1}: {palabras_clave}")

# Asignar el tópico con mayor probabilidad para cada tweet
def obtener_topico_predominante(tweet_bow):
    topicos_tweet = modelo_lda[tweet_bow]
    # Ordenar los tópicos por probabilidad y devolver el índice del tópico con mayor probabilidad
    return max(topicos_tweet, key=lambda x: x[1])[0]

# Crear una lista con los tópicos predominantes para cada tweet
topicos_predominantes = [obtener_topico_predominante(tweet_bow) for tweet_bow in corpus]

# Agregar la nueva columna al DataFrame
# df_200['Dominant_Topic'] = topicos_predominantes

# Guardar el DataFrame de los primeros 200 tweets en un nuevo archivo CSV
# df_200.to_csv('TPI/databases/TwitterData_200_with_topics.csv', index=False)

print(topicos_predominantes)
# print("Archivo guardado correctamente con los tópicos añadidos.")