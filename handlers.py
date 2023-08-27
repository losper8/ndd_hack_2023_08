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
        df_answer, ans, df = find_similar_questions(msg.text)

        df_question = df['QUESTION'].to_list()

        kb = [[types.KeyboardButton(text=question)]
              for question in df_question]
        keyboard = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            is_persistent=True,
            input_field_placeholder = "Похожие вопросы"
        )
        if df_answer == 'EMP':
            await msg.answer(f"{ans}", reply_markup=keyboard)
        else:
            await msg.answer(f"{df_answer[0]['summary_text']}", reply_markup=keyboard)
    except:
        await msg.answer("Что-то пошло не так")

    kb = []
