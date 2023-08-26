from deeppavlov import build_model
from main import extract_answers
model = build_model('squad_ru_bert', download=True)


print(model(['DeepPavlov is a library for NLP and dialog systems.'], ['What is DeepPavlov?']))