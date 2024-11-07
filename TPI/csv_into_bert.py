import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
D = pd.read_csv('TPI/databases/Twitterdatainsheets.csv')

# Cargar el tokenizador y el modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
textos = D['text'].tolist()

# Obtener los embeddings para cada texto
batch_size = 8
embeddings_oraciones = []

# Procesar en lotes
for i in range(0, len(textos), batch_size):
    batch_texts = textos[i:i + batch_size]  # Obtener el lote actual de textos
    tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        salida = model(**tokens)

    # Extraer los embeddings de [CLS] y agregar al listado de embeddings
    embeddings_oraciones.append(salida.last_hidden_state[:, 0, :])

# Concatenar todos los embeddings
embeddings_oraciones = torch.cat(embeddings_oraciones, dim=0)
print(embeddings_oraciones)
D.to_csv("archivo_con_embeddings.csv", index=False)
