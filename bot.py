import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any

import openai
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
        # Clean up old messages
        self.user_messages[user_id] = [
            timestamp for timestamp in self.user_messages[user_id]
            if current_time - timestamp < timedelta(days=1)
        ]
        return len(self.user_messages[user_id]) < self.max_daily_messages
    
    def add_message(self, user_id: int) -> None:
        self.user_messages[user_id].append(datetime.now())
    
    def get_remaining_messages(self, user_id: int) -> int:
        self.can_send_message(user_id)  # Clean up old messages
        return self.max_daily_messages - len(self.user_messages[user_id])

class DocumentStore:
    def __init__(self, pdf_directory: str = "./pdfs"):
        self.pdf_directory = pdf_directory
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store = None
            self.pdf_files = []  # Track loaded PDFs
            self.initialize_vector_store()
            logger.info(f"DocumentStore initialized with directory: {pdf_directory}")
        except Exception as e:
            logger.error(f"Error initializing DocumentStore: {str(e)}")
            raise

    def read_pdf(self, pdf_path: str) -> str:
        """Read and extract text from a PDF file."""
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
        """Return list of currently loaded PDF files."""
        try:
            return [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        except Exception as e:
            logger.error(f"Error listing PDFs: {str(e)}")
            return []

    def initialize_vector_store(self) -> None:
        """Initialize the vector store with PDF documents."""
        try:
            if not os.path.exists(self.pdf_directory):
                os.makedirs(self.pdf_directory)
                logger.info(f"Created PDF directory: {self.pdf_directory}")

            self.pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            
            if not self.pdf_files:
                logger.warning("⚠️ No PDF files found in directory. Please add PDFs to the 'pdfs' folder.")
                print("⚠️ No PDF files found. Please add PDFs to the 'pdfs' folder.")
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
                logger.info(f"✅ Vector store initialized with {len(texts)} text chunks from {len(self.pdf_files)} PDFs")
                print(f"✅ Successfully loaded {len(self.pdf_files)} PDFs with {len(texts)} text chunks")
            else:
                logger.warning("No valid text content found in PDFs")
                print("⚠️ No valid text content found in PDFs")
        
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            print(f"❌ Error initializing vector store: {str(e)}")
            raise

    def query_documents(self, query: str, k: int = 3) -> str:
        """Query the vector store for relevant document content."""
        try:
            if not self.pdf_files:
                return "No PDF documents have been loaded yet. Please add PDFs to the 'pdfs' folder."

            if self.vector_store is None:
                return "Vector store not initialized. No documents available for searching."
            
            relevant_docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not relevant_docs_with_scores:
                return "No relevant information found in the loaded documents."

            formatted_responses = []
            for doc, score in relevant_docs_with_scores:
                if score < 0.5:  # Lower score means better match
                    source = doc.metadata.get("source", "Unknown source")
                    formatted_responses.append(
                        f"From {source}:\n{doc.page_content.strip()}"
                    )

            if formatted_responses:
                context = "\n\n".join(formatted_responses)
                logger.info(f"Found {len(formatted_responses)} relevant contexts for query")
                return context
            else:
                return "Found some documents, but they weren't relevant enough to the query."

        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return f"Error querying documents: {str(e)}"

    async def reload_pdfs(self) -> str:
        """Reload all PDFs in the directory."""
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

class TelegramGPTBot:
    def __init__(self, telegram_token: str, openai_api_key: str, max_daily_messages: int = 5):
        self.telegram_token = telegram_token
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.doc_store = DocumentStore()
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter(max_daily_messages=max_daily_messages)
        self.application = None

    async def setup_commands(self):
        """Set up the bot commands in Telegram."""
        commands = [
            ("start", "Start the bot"),
            ("help", "Show help information"),
            ("clear", "Clear conversation history"),
            ("quota", "Check remaining daily messages"),
            ("pdfs", "List loaded PDF documents"),
            ("reload", "Reload PDF documents")
        ]
        
        try:
            await self.application.bot.set_my_commands(commands)
            logger.info("Bot commands have been set up successfully")
        except Exception as e:
            logger.error(f"Failed to set up bot commands: {e}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
      user_id = update.effective_user.id
      remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
      
      pdf_status = f"📚 Loaded PDFs: {len(self.doc_store.pdf_files)}" if self.doc_store.pdf_files else "⚠️ No PDFs loaded"
      
      welcome_message = f"""
      👋 Welcome! I'm your AI assistant with access to both GPT-4 and a knowledge base of documents.
      
      {pdf_status}
      You have {remaining_messages} messages remaining for today.
      
      Commands:
      /start - Show this message
      /help - Show help information
      /clear - Clear your conversation history
      /quota - Check your remaining daily messages
      /pdfs - List loaded PDF documents
      /reload - Reload PDF documents
      
      To get started, make sure there are PDF files in the 'pdfs' directory!
      """
      await update.message.reply_text(welcome_message)
      logger.info(f"New user started the bot: {user_id}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
        
        help_text = f"""
        I can help you with:
        1. General questions using GPT-4
        2. Questions about the loaded PDF documents
        3. Maintaining context of our conversation

        Daily Usage:
        - You have {remaining_messages} messages remaining today
        - The quota resets every 24 hours
        
        Commands:
        /start - Start the bot
        /help - Show this help message
        /clear - Clear conversation history
        /quota - Check remaining messages
        /pdfs - List loaded PDF documents
        /reload - Reload PDF documents
        """
        await update.message.reply_text(help_text)

    async def list_pdfs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List all currently loaded PDFs."""
        pdfs = self.doc_store.get_loaded_pdfs()
        if pdfs:
            pdf_list = "\n".join(f"📄 {pdf}" for pdf in pdfs)
            await update.message.reply_text(f"Loaded PDFs:\n\n{pdf_list}")
        else:
            await update.message.reply_text("No PDFs currently loaded.")

    async def reload_pdfs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reload all PDFs from the directory."""
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
                    "❌ You've reached your daily message limit. "
                    "Please try again tomorrow or use /quota to check your status."
                )
                logger.info(f"Rate limit reached for user: {user_id}")
                return

            user_message = update.message.text
            logger.info(f"Received message from user {user_id}: {user_message[:100]}...")

            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, 
                action='typing'
            )

            relevant_context = self.doc_store.query_documents(user_message)
            
            if relevant_context:
                system_message = f"""You are a helpful assistant with access to specific document knowledge.
                Please answer the question using the following relevant information from the documents:
                
                {relevant_context}
                
                If you use information from the documents, mention the source.
                If the provided context isn't sufficient to fully answer the question, 
                you can supplement with your general knowledge but make it clear which parts
                are from the documents and which are from your general knowledge."""
            else:
                system_message = """You are a helpful assistant. Please note that while I can access PDF documents,
                I couldn't find relevant information in the loaded documents for this specific query.
                I'll answer based on my general knowledge."""

            messages = [
                {"role": "system", "content": system_message}
            ]

            messages.extend(self.conversation_manager.get_conversation_history(user_id))
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                temperature=0.1
            )

            ai_response = response.choices[0].message.content

            self.rate_limiter.add_message(user_id)
            self.conversation_manager.add_message(user_id, "user", user_message)
            self.conversation_manager.add_message(user_id, "assistant", ai_response)

            remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
            
            full_response = f"{ai_response}\n\n📊 You have {remaining_messages} messages remaining today."
            await update.message.reply_text(full_response)
            logger.info(f"Sent response to user {user_id}, {remaining_messages} messages remaining")

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
          self.application.add_handler(CommandHandler("help", self.help_command))
          self.application.add_handler(CommandHandler("clear", self.clear_history))
          self.application.add_handler(CommandHandler("quota", self.check_quota))
          self.application.add_handler(CommandHandler("pdfs", self.list_pdfs))
          self.application.add_handler(CommandHandler("reload", self.reload_pdfs))
          self.application.add_handler(MessageHandler(
              filters.TEXT & ~filters.COMMAND, 
              self.handle_message
          ))

          # Setup commands in Telegram and start the bot
          logger.info("Bot started")
          self.application.run_polling(allowed_updates=Update.ALL_TYPES)

      except Exception as e:
          logger.error(f"Failed to start bot: {str(e)}")
          raise

def validate_environment() -> bool:
    """Validate all required environment variables are set."""
    required_vars = {
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
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
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        MAX_DAILY_MESSAGES = int(os.getenv('MAX_DAILY_MESSAGES', '5'))

        # Initialize and run bot
        logger.info("Initializing bot...")
        bot = TelegramGPTBot(TELEGRAM_TOKEN, OPENAI_API_KEY, MAX_DAILY_MESSAGES)
        logger.info("Starting bot...")
        bot.run()

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise