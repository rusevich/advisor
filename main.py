from dotenv import load_dotenv
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
# Load environment variables from .env file
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', "")

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')

app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(CommandHandler("hello", hello))

app.run_polling()

if __name__ == "__main__":
    pass