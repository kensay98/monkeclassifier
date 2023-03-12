import logging

from fastai.vision.all import load_learner
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters, CommandHandler,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)
logging.getLogger('hpack').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

learn = load_learner('models/export.pkl')
TELEGRAM_TOKEN = '488312680:AAGsKHKufV9TQEAB8-g6INps-W82G_noRP8'


def get_prediction(path):
    monke, _, probs = learn.predict(path)
    return monke, max(probs)


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo the user message."""
    file_id = update.message.photo[-1].file_id

    file = await context.bot.get_file(file_id)

    file = await file.download_to_drive('image.jpg')

    prediction, prob = get_prediction('image.jpg')

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"This is {prediction} with {prob * 100:.2f}% probability"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hi! I'm a bot that can tell you what kind of monkey is in a photo. Send me a photo and I'll tell you!"
    )


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(MessageHandler(filters.PHOTO, photo))
    application.add_handler(CommandHandler("start", start))

    # show_data_handler = CommandHandler("show_data", show_data)
    # application.add_handler(show_data_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
