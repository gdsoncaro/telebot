import os
import logging
from datetime import datetime
from typing import List, Dict

import openai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        
    def add_message(self, user_id: int, role: str, content: str) -> None:
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message = {"role": role, "content": content}
        self.conversations[user_id].append(message)
        if len(self.conversations[user_id]) > 10:  # Keep last 10 messages
            self.conversations[user_id].pop(0)
    
    def get_conversation_history(self, user_id: int) -> List[Dict[str, str]]:
        return self.conversations.get(user_id, [])

class TelegramGPTBot:
    def __init__(self, telegram_token: str, openai_api_key: str):
        self.telegram_token = telegram_token
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.conversation_manager = ConversationManager()
        self.application = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        welcome_message = """
        👋 Hello! I'm your chat assistant. Feel free to ask me anything!
        
        Commands:
        /start - Show this message
        /clear - Clear your conversation history
        """
        await update.message.reply_text(welcome_message)
        logger.info(f"New user started the bot: {update.effective_user.id}")

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        self.conversation_manager.conversations[user_id] = []
        await update.message.reply_text("Conversation history cleared! 🧹")
        logger.info(f"Conversation history cleared for user: {user_id}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_id = update.effective_user.id
            user_message = update.message.text
            
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, 
                action='typing'
            )

            messages = [
                {"role": "developer", "content": "You are a helpful assistant."}
            ]

            messages.extend(self.conversation_manager.get_conversation_history(user_id))
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="o1-2024-12-17",
                messages=messages,
                max_completion_tokens=500
            )

            ai_response = response.choices[0].message.content

            self.conversation_manager.add_message(user_id, "user", user_message)
            self.conversation_manager.add_message(user_id, "assistant", ai_response)

            await update.message.reply_text(ai_response)
            logger.info(f"Sent response to user {user_id}")

        except Exception as e:
            error_message = f"Sorry, an error occurred: {str(e)}"
            await update.message.reply_text(error_message)
            logger.error(f"Error handling message from user {user_id}: {str(e)}")

    def run(self) -> None:
        """Start the bot."""
        try:
            self.application = Application.builder().token(self.telegram_token).build()

            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("clear", self.clear_history))
            self.application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, 
                self.handle_message
            ))

            logger.info("Bot started")
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)

        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}")
            raise

if __name__ == '__main__':
    try:
        # Load configuration from environment variables
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN_CLAUDE')
        OPENAI_API_KEY = os.getenv('OPENAI_API_01_KEY')

        if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
            raise ValueError("Missing required environment variables. Please check your .env file.")

        # Initialize and run bot
        logger.info("Initializing bot...")
        bot = TelegramGPTBot(TELEGRAM_TOKEN, OPENAI_API_KEY)
        logger.info("Starting bot...")
        bot.run()

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise