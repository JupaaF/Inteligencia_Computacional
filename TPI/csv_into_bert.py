import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import csv
D = pd.read_csv('TPI/databases/engagement_info_actualizado.csv')

# Cargar modelo y tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

batch_size = 256

csv_filename = "embeddings_BERT.csv"

# Abrir el archivo en modo escritura
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Procesar y guardar cada embedding
    with torch.no_grad():  # Desactiva cálculo de gradientes
        for i in range(0, len(D), batch_size):
            # Seleccionar un lote del DataFrame
            batch = D.iloc[i:i + batch_size]
            texts = batch[" text"].tolist()

            # Tokenizar y generar embeddings
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Embedding de [CLS]

            # Escribir los embeddings al archivo CSV
            writer.writerows(batch_embeddings)

print(f"Embeddings guardados en '{csv_filename}'")

