import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from transformers import pipeline
df = pd.read_pickle("emb_sn-xlm-roberta-base-snli-mnli-anli-xnli.pkl")

model_name = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"
model = SentenceTransformer(model_name)  # вызов класса
embeddings = df["эмбеддинги"].tolist()


def find_similar_questions(question):
    not_found_m = 'По вашему запросу ничего не найдено'
    # Вычисление эмбеддинга введенного вопроса
    question_embedding = model.encode([question])

    # Вычисление сходства между введенным вопросом и всеми вопросами в столбце эмбеддингов
    distances = cosine_distances(question_embedding, embeddings)

    # Поиск индексов топ-5 наименее удаленных вопросов
    top_indices = distances.argsort()[0][:5]

    top_distances = distances[0][top_indices]
    # Получение топ-5 наиболее похожих вопросов и соответствующих ответов
    similar_questions = df.loc[top_indices, ['QUESTION', 'ANSWER']]

    if top_distances[0] > 0.5:
        return 'EMP', not_found_m, similar_questions

    ans = similar_questions.iloc[0]['ANSWER']
    sum_ans = 'EMP'
    if len(ans) > 120:
        pipe = pipeline('summarization', model='d0rj/rut5-base-summ')
        sum_ans = pipe(ans)
    return  sum_ans, ans, similar_questions
