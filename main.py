from dotenv import load_dotenv
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from rag_retrieve import retrieve_5_most_relevant

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# Load environment variables from .env file
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

model = OpenRouterModel(
    'deepseek/deepseek-chat',
    provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY),
)

agent = Agent(
    model,
    system_prompt=(
        "You are Niccolò Machiavelli. Answer in this exact structure:\n"
        "1. Citation from Machiavelli's The Prince.\n"
        "2. Interpretation of the citation.\n"
        "3. Direct steps the user should take.\n"
        "Use the provided excerpts as your source material. Answer in Russian."
    ),
)

def ask_advice(user_input):
    passages = retrieve_5_most_relevant(user_input)

    context = "\n\n".join(
        f"{chapter}\n{paragraph}" for chapter, paragraph in passages
    )

    prompt = f"""Situation:
    {user_input}
    
    Relevant excerpts from The Prince:
    {context}
    """

    return agent.run_sync(prompt).output

user_input = input("? ")
ask_advice(user_input)

#
# async def machiavelli_advice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     await update.message.reply_text(f'{update.effective_user.first_name}')
#
# app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
# app.add_handler(CommandHandler("advice", hello))
#
# app.run_polling()