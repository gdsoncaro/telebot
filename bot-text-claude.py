import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any

from anthropic import Anthropic
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='bot.log'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, max_daily_messages: int):
        self.max_daily_messages = max_daily_messages
        self.user_messages: Dict[int, List[datetime]] = defaultdict(list)
    
    def can_send_message(self, user_id: int) -> bool:
        current_time = datetime.now()
        self.user_messages[user_id] = [
            timestamp for timestamp in self.user_messages[user_id]
            if current_time - timestamp < timedelta(days=1)
        ]
        return len(self.user_messages[user_id]) < self.max_daily_messages
    
    def add_message(self, user_id: int) -> None:
        self.user_messages[user_id].append(datetime.now())
    
    def get_remaining_messages(self, user_id: int) -> int:
        self.can_send_message(user_id)
        return self.max_daily_messages - len(self.user_messages[user_id])

class DocumentStore:
    def __init__(self, pdf_directory: str = "./pdfs"):
        self.pdf_directory = pdf_directory
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store = None
            self.pdf_files = []
            self.initialize_vector_store()
            logger.info(f"DocumentStore initialized with directory: {pdf_directory}")
        except Exception as e:
            logger.error(f"Error initializing DocumentStore: {str(e)}")
            raise

    def read_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                logger.info(f"Successfully read PDF: {pdf_path}")
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def get_loaded_pdfs(self) -> List[str]:
        try:
            return [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        except Exception as e:
            logger.error(f"Error listing PDFs: {str(e)}")
            return []

    def initialize_vector_store(self) -> None:
        try:
            if not os.path.exists(self.pdf_directory):
                os.makedirs(self.pdf_directory)
                logger.info(f"Created PDF directory: {self.pdf_directory}")

            self.pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            
            if not self.pdf_files:
                logger.warning("⚠️ No PDF files found in directory.")
                print("⚠️ No PDF files found in directory.")
                return

            documents = []
            for filename in self.pdf_files:
                file_path = os.path.join(self.pdf_directory, filename)
                text = self.read_pdf(file_path)
                if text:
                    logger.info(f"Successfully loaded PDF: {filename}")
                    print(f"📚 Loaded PDF: {filename}")
                    doc = Document(
                        page_content=text,
                        metadata={"source": filename}
                    )
                    documents.append(doc)
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                )
                texts = text_splitter.split_documents(documents)
                
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
                logger.info(f"✅ Vector store initialized with {len(texts)} text chunks")
                print(f"✅ Successfully loaded {len(self.pdf_files)} PDFs with {len(texts)} text chunks")
            else:
                logger.warning("No valid text content found in PDFs")
                print("⚠️ No valid text content found in PDFs")
        
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            print(f"❌ Error initializing vector store: {str(e)}")
            raise

    def query_documents(self, query: str, k: int = 3) -> str:
        try:
            if not self.pdf_files:
                return "No PDF documents have been loaded yet."

            if self.vector_store is None:
                return "Vector store not initialized."
            
            relevant_docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not relevant_docs_with_scores:
                return "No relevant information found."

            formatted_responses = []
            for doc, score in relevant_docs_with_scores:
                if score < 0.5:
                    source = doc.metadata.get("source", "Unknown source")
                    formatted_responses.append(
                        f"From {source}:\n{doc.page_content.strip()}"
                    )

            if formatted_responses:
                return "\n\n".join(formatted_responses)
            else:
                return "No sufficiently relevant information found."

        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return f"Error querying documents: {str(e)}"

    async def reload_pdfs(self) -> str:
        try:
            self.vector_store = None
            self.initialize_vector_store()
            pdf_count = len(self.get_loaded_pdfs())
            return f"Successfully reloaded {pdf_count} PDFs."
        except Exception as e:
            error_msg = f"Error reloading PDFs: {str(e)}"
            logger.error(error_msg)
            return error_msg

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[int, List[Dict[str, Any]]] = {}
        
    def add_message(self, user_id: int, role: str, content: str) -> None:
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversations[user_id].append(message)
        if len(self.conversations[user_id]) > 10:
            self.conversations[user_id].pop(0)
    
    def get_conversation_history(self, user_id: int) -> List[Dict[str, str]]:
        if user_id not in self.conversations:
            return []
        return [{"role": msg["role"], "content": msg["content"]} 
                for msg in self.conversations[user_id]]

class TelegramClaudeBot:
    def __init__(self, telegram_token: str, anthropic_api_key: str, max_daily_messages: int = 5):
        self.telegram_token = telegram_token
        self.client = Anthropic(api_key=anthropic_api_key)
        self.doc_store = DocumentStore()
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter(max_daily_messages=max_daily_messages)
        self.application = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
        
        pdf_status = f"📚 PDFs Loaded: {len(self.doc_store.pdf_files)}" if self.doc_store.pdf_files else "⚠️ No PDFs loaded"
        
        welcome_message = f"""
        👋 Hello! You can ask me anything about the loaded PDFs.
        
        {pdf_status}
        Messages remaining: {remaining_messages}
        
        Commands:
        /start - Show this message
        /clear - Clear conversation history
        /quota - Check remaining messages
        /pdfs - List loaded PDFs
        /reload - Reload PDFs
        """
        await update.message.reply_text(welcome_message)
        logger.info(f"New user started the bot: {user_id}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
        
        help_text = f"""
        I can help you with:
        1. Questions about the loaded PDF documents
        2. Maintaining context of our conversation

        You have {remaining_messages} messages remaining today
        
        Commands:
        /start - Start the bot
        /help - Show this help message
        /clear - Clear conversation history
        /quota - Check remaining messages
        /pdfs - List loaded PDFs
        /reload - Reload PDFs
        """
        await update.message.reply_text(help_text)

    async def list_pdfs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        pdfs = self.doc_store.get_loaded_pdfs()
        if pdfs:
            pdf_list = "\n".join(f"📄 {pdf}" for pdf in pdfs)
            await update.message.reply_text(f"Loaded PDFs:\n\n{pdf_list}")
        else:
            await update.message.reply_text("No PDFs currently loaded.")

    async def reload_pdfs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Reloading PDFs... Please wait.")
        result = await self.doc_store.reload_pdfs()
        await update.message.reply_text(result)

    async def check_quota(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
        
        quota_message = f"""
        Daily Message Quota Status:
        
        ▫️ Remaining messages: {remaining_messages}
        ▫️ Maximum daily limit: {self.rate_limiter.max_daily_messages}
        
        The quota resets 24 hours after each message is sent.
        """
        await update.message.reply_text(quota_message)

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        self.conversation_manager.conversations[user_id] = []
        await update.message.reply_text("Conversation history cleared! 🧹")
        logger.info(f"Conversation history cleared for user: {user_id}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_id = update.effective_user.id
            
            if not self.rate_limiter.can_send_message(user_id):
                await update.message.reply_text(
                    "❌ Daily message limit reached. Try again tomorrow or use /quota to check status."
                )
                return

            user_message = update.message.text
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, 
                action='typing'
            )

            relevant_context = self.doc_store.query_documents(user_message)
            
            if relevant_context:
                system_message = f"""You are a helpful assistant with access to document knowledge.
                Answer based on this information from the documents:
                
                {relevant_context}
                
                Cite sources when using document information."""
            else:
                system_message = "You are a helpful assistant. I couldn't find relevant information in the documents for this query."

            # Convert conversation history to Claude's format
            messages = []
            history = self.conversation_manager.get_conversation_history(user_id)
            
            # Add system message and conversation history
            messages.append({"role": "user", "content": f"System: {system_message}"})
            for msg in history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})

            messages.append({"role": "user", "content": user_message})

            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=750,
                temperature=0.1,
                messages=messages
            )

            ai_response = response.content[0].text

            self.rate_limiter.add_message(user_id)
            self.conversation_manager.add_message(user_id, "user", user_message)
            self.conversation_manager.add_message(user_id, "assistant", ai_response)

            remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
            
            full_response = f"{ai_response}\n\n📊 {remaining_messages} messages remaining today."
            await update.message.reply_text(full_response)
            logger.info(f"Response sent to user {user_id}, {remaining_messages} messages remaining")

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            await update.message.reply_text(error_message)
            logger.error(f"Error handling message from user {user_id}: {str(e)}")

    def run(self) -> None:
        try:
            self.application = Application.builder().token(self.telegram_token).build()

            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("clear", self.clear_history))
            self.application.add_handler(CommandHandler("quota", self.check_quota))
            self.application.add_handler(CommandHandler("pdfs", self.list_pdfs))
            self.application.add_handler(CommandHandler("reload", self.reload_pdfs))
            self.application.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, 
                self.handle_message
            ))

            logger.info("Bot started")
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)

        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}")
            raise

def validate_environment() -> bool:
    """Validate all required environment variables are set."""
    required_vars = {
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN_CLAUDE'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

if __name__ == '__main__':
    try:
        # Validate environment
        if not validate_environment():
            raise ValueError("Missing required environment variables. Please check your .env file.")

        # Load configuration from environment variables
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN_CLAUDE')
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        MAX_DAILY_MESSAGES = int(os.getenv('MAX_DAILY_MESSAGES', '5'))

        # Initialize and run bot
        logger.info("Initializing bot...")
        bot = TelegramClaudeBot(TELEGRAM_TOKEN, ANTHROPIC_API_KEY, MAX_DAILY_MESSAGES)
        logger.info("Starting bot...")
        bot.run()

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise