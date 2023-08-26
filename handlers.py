from aiogram import Router, types
from aiogram.types import Message
from aiogram.filters import Command
from fsq import find_similar_questions

router = Router()


@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer("Привет! Я твой МФЦ помошник. Для начала работы введите интересующий вас вопрос!")

# TODO message from user and response


@router.message()
async def message_handler(msg: Message):
    try:
        df = find_similar_questions(msg.text)
        df_answer = df['ANSWER'].to_list()
        df_question = df['QUESTION'].to_list()
        kb = [[types.KeyboardButton(text=question)]
              for question in df_question]
        keyboard = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            is_persistent=True,
            input_field_placeholder="Похожие вопросы"
        )
        for item in df_answer:
            await msg.answer(f"{item}", reply_markup=keyboard)
    except:
        await msg.answer("Ой ой млшики обосрались")

    kb = []
