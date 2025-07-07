import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import mercadopago
from dataclasses import dataclass
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

@dataclass
class Payment:
    preference_id: str
    created_at: datetime
    status: str = "pending"
    messages_added: bool = False

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
                logger.warning("No PDF files found in directory.")
                return

            documents = []
            for filename in self.pdf_files:
                file_path = os.path.join(self.pdf_directory, filename)
                text = self.read_pdf(file_path)
                if text:
                    logger.info(f"Successfully loaded PDF: {filename}")
                    doc = Document(
                        page_content=text,
                        metadata={"source": filename}
                    )
                    documents.append(doc)

            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                texts = text_splitter.split_documents(documents)
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
                logger.info(f"Vector store initialized with {len(texts)} text chunks")

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
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
                if score < 0.5:  # Only include relevant matches
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

class PaymentManager:
    def __init__(self, access_token: str):
        self.sdk = mercadopago.SDK(access_token)
        self.payments: Dict[int, List[Payment]] = defaultdict(list)
        
    def create_qr_payment(self, user_id: int) -> Tuple[str, str]:
        try:
            preference_data = {
                "items": [{
                    "title": "4 Messages for Claude Bot",
                    "quantity": 1,
                    "currency_id": "ARS",
                    "unit_price": 300
                }],
                "external_reference": f"user_{user_id}",
                "notification_url": "https://your-webhook-url.com/notifications",
                "payment_methods": {
                    "excluded_payment_methods": [],
                    "excluded_payment_types": []
                }
            }
            
            preference_response = self.sdk.preference().create(preference_data)
            
            if preference_response["status"] == 201:
                preference_id = preference_response["response"]["id"]
                qr_data = self.sdk.qr().create(preference_id)
                
                # Store payment info
                self.payments[user_id].append(
                    Payment(
                        preference_id=preference_id,
                        created_at=datetime.now()
                    )
                )
                
                return preference_id, qr_data["response"]["qr_data"]
            else:
                raise Exception("Failed to create payment preference")
                
        except Exception as e:
            logger.error(f"Error creating payment: {str(e)}")
            raise
            
    def check_payment_status(self, user_id: int, preference_id: str) -> Tuple[str, bool]:
        try:
            # Search for payments associated with this preference
            search_params = {"preference_id": preference_id}
            payment_response = self.sdk.payment().search(search_params)
            
            if payment_response["status"] == 200 and payment_response["response"]["results"]:
                # Get the latest payment for this preference
                payment = payment_response["response"]["results"][0]
                status = payment["status"]
                
                # Find this payment in our records
                for stored_payment in self.payments[user_id]:
                    if stored_payment.preference_id == preference_id:
                        stored_payment.status = status
                        # If payment is approved and messages haven't been added yet
                        if status == "approved" and not stored_payment.messages_added:
                            stored_payment.messages_added = True
                            return status, True
                        return status, False
                        
            return "pending", False
            
        except Exception as e:
            logger.error(f"Error checking payment status: {str(e)}")
            return "error", False

class MessageCounter:
    def __init__(self, messages_per_payment: int = 4):
        self.messages_per_payment = messages_per_payment
        self.user_messages: Dict[int, int] = defaultdict(int)
        
    def can_send_message(self, user_id: int) -> bool:
        return self.user_messages[user_id] > 0
    
    def add_messages(self, user_id: int) -> None:
        self.user_messages[user_id] += self.messages_per_payment
        
    def use_message(self, user_id: int) -> None:
        if self.user_messages[user_id] > 0:
            self.user_messages[user_id] -= 1
            
    def get_remaining_messages(self, user_id: int) -> int:
        return self.user_messages[user_id]

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
    def __init__(
        self, 
        telegram_token: str, 
        anthropic_api_key: str, 
        mercadopago_token: str,
        initial_messages: int = 3
    ):
        self.telegram_token = telegram_token
        self.client = Anthropic(api_key=anthropic_api_key)
        self.doc_store = DocumentStore()
        self.conversation_manager = ConversationManager()
        self.payment_manager = PaymentManager(mercadopago_token)
        self.message_counter = MessageCounter()
        self.application = None
        self.initial_messages = initial_messages

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        
        # Give initial messages to new users
        if self.message_counter.get_remaining_messages(user_id) == 0:
            self.message_counter.user_messages[user_id] = self.initial_messages
        
        remaining_messages = self.message_counter.get_remaining_messages(user_id)
        
        welcome_message = f"""
        👋 Hello! You can ask me anything about the loaded PDFs.
        
        📚 PDFs Loaded: {len(self.doc_store.pdf_files) if self.doc_store.pdf_files else "No PDFs loaded"}
        ✉️ Messages remaining: {remaining_messages}
        
        When you run out of messages, you'll be prompted to pay 300 ARS for 4 more messages.
        
        Commands:
        /start - Show this message
        /clear - Clear conversation history
        /messages - Check remaining messages
        /pdfs - List loaded PDFs
        /reload - Reload PDFs
        /check_payment - Check status of pending payment
        """
        await update.message.reply_text(welcome_message)

    async def check_payment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        
        if not self.payment_manager.payments[user_id]:
            await update.message.reply_text("No pending payments found.")
            return
            
        latest_payment = self.payment_manager.payments[user_id][-1]
        status, messages_added = self.payment_manager.check_payment_status(
            user_id, 
            latest_payment.preference_id
        )
        
        if messages_added:
            self.message_counter.add_messages(user_id)
            await update.message.reply_text(
                f"✅ Payment approved! Added 4 messages to your account.\n"
                f"You now have {self.message_counter.get_remaining_messages(user_id)} messages remaining."
            )
        else:
            status_messages = {
                "approved": "✅ Payment already processed.",
                "pending": "⏳ Payment pending. Please complete the payment.",
                "rejected": "❌ Payment rejected. A new QR code will be generated when you run out of messages.",
                "error": "⚠️ Error checking payment status. Please try again later.",
                "unknown": "⚠️ Unknown payment status. Please contact support."
            }
            await update.message.reply_text(status_messages.get(status, "Unknown status"))

    async def check_messages(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        remaining = self.message_counter.get_remaining_messages(user_id)
        await update.message.reply_text(f"You have {remaining} messages remaining.")

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        self.conversation_manager.conversations[user_id] = []
        await update.message.reply_text("Conversation history cleared! 🧹")

    async def list_pdfs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        pdfs = self.doc_store.get_loaded_pdfs()
        if pdfs:
            pdf_list = "\n".join(f"📄 {pdf}" for pdf in pdfs)
            await update.message.reply_text(f"Loaded PDFs:\n\n{pdf_list}")
        else:
            await update.message.reply_text("No PDFs currently loaded.")

    async def reload_pdfs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Reloading PDFs... Please wait.")
        try:
            self.doc_store.initialize_vector_store()
            await update.message.reply_text(f"Successfully reloaded {len(self.doc_store.pdf_files)} PDFs.")
        except Exception as e:
            await update.message.reply_text(f"Error reloading PDFs: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_id = update.effective_user.id
            
            if not self.message_counter.can_send_message(user_id):
                try:
                    preference_id, qr_code = self.payment_manager.create_qr_payment(user_id)
                    
                    message = """
                    ⭐️ Get More Messages!
                    
                    You've used all your messages. Scan this QR code to get 4 more messages:
                    • Price: 300 ARS
                    • Messages: 4
                    
                    Use /check_payment after paying to refresh your message count.
                    """
                    
                    await update.message.reply_text(message)
                    await update.message.reply_photo(qr_code)
                    return
                    
                except Exception as e:
                    await update.message.reply_text(f"Error generating payment: {str(e)}")
                    return
            # Continuing from handle_message method
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
                messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": user_message})

            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-7-sonnet",
                max_tokens=750,
                temperature=0.1,
                messages=messages
            )

            ai_response = response.content[0].text

            # Record the interaction
            self.message_counter.use_message(user_id)
            self.conversation_manager.add_message(user_id, "user", user_message)
            self.conversation_manager.add_message(user_id, "assistant", ai_response)

            # Get remaining messages
            remaining_messages = self.message_counter.get_remaining_messages(user_id)
            
            full_response = f"{ai_response}\n\n📊 {remaining_messages} messages remaining."
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
            self.application.add_handler(CommandHandler("messages", self.check_messages))
            self.application.add_handler(CommandHandler("pdfs", self.list_pdfs))
            self.application.add_handler(CommandHandler("reload", self.reload_pdfs))
            self.application.add_handler(CommandHandler("check_payment", self.check_payment))
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
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN_MP'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'MERCADOPAGO_ACCESS_TOKEN': os.getenv('MERCADOPAGO_ACCESS_TOKEN')
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
            raise ValueError("Missing required environment variables")

        # Load configuration
        TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN_MP')
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        MERCADOPAGO_TOKEN = os.getenv('MERCADOPAGO_ACCESS_TOKEN')
        INITIAL_MESSAGES = int(os.getenv('FREE_MESSAGES', '3'))

        # Initialize and run bot
        logger.info("Initializing bot...")
        bot = TelegramClaudeBot(
            TELEGRAM_TOKEN, 
            ANTHROPIC_API_KEY,
            MERCADOPAGO_TOKEN,
            INITIAL_MESSAGES
        )
        logger.info("Starting bot...")
        bot.run()

    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise