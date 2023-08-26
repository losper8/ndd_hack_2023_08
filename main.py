import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

df = pd.read_pickle("emb_sn-xlm-roberta-base-snli-mnli-anli-xnli.pkl")

model_name = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"
model = SentenceTransformer(model_name) # вызов класса 
embeddings = df["эмбеддинги"].tolist()
def find_similar_questions(question):
    # Вычисление эмбеддинга введенного вопроса
    question_embedding = model.encode([question])
    
    # Вычисление сходства между введенным вопросом и всеми вопросами в столбце эмбеддингов
    distances = cosine_distances(question_embedding, embeddings)
    
    # Поиск индексов топ-5 наименее удаленных вопросов
    top_indices = distances.argsort()[0][:5]
    
    # Получение топ-5 наиболее похожих вопросов и соответствующих ответов
    similar_questions = df.loc[top_indices, ['QUESTION', 'ANSWER']]
    
    return similar_questions

# Ввод вопроса
user_question = input("Введите ваш вопрос: ")

# Поиск топ-5 наиболее похожих вопросов
similar_questions = find_similar_questions(user_question)

# Вывод результатов в формате датафрейма
print(similar_questions)