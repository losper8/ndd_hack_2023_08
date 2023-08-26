import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
df = pd.read_excel('train_dataset_Датасет.xlsx')
print(df.iloc[0])

sentences = ["This is an example sentence", "Each sentence is converted"]
model_name = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"
model = SentenceTransformer(model_name)  # вызов класса


# Вычисление эмбеддингов для вопросов в столбце 'вопрос'
embeddings = model.encode(df['QUESTION'])

# Функция для преобразования каждого эмбеддинга в одномерный массив


def flatten_embedding(embedding):
    return np.ravel(embedding)


# Преобразование каждого эмбеддинга в одномерный массив
embeddings_flat = np.apply_along_axis(flatten_embedding, 1, embeddings)

# Добавление столбца 'эмбеддинги' в датафрейм
df['эмбеддинги'] = embeddings_flat.tolist()

df.to_pickle("emb_sn-xlm-roberta-base-snli-mnli-anli-xnli.pkl")

# Вывод датафрейма с добавленным столбцом эмбеддингов
