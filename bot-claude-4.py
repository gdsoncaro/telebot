import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any

from anthropic import Anthropic
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
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
    def __init__(self, base_directory: str = "./"):
        self.base_directory = base_directory
        self.current_directory = None
        self.selected_pdfs = []
        try:
            self.embeddings = OpenAIEmbeddings()
            self.vector_store = None
            self.pdf_files = []
            logger.info(f"DocumentStore initialized with base directory: {base_directory}")
        except Exception as e:
            logger.error(f"Error initializing DocumentStore: {str(e)}")
            raise

    def get_pdf_directories(self) -> List[str]:
        """Get all directories that start with 'pdfs-' in the base directory."""
        try:
            directories = []
            for item in os.listdir(self.base_directory):
                item_path = os.path.join(self.base_directory, item)
                if os.path.isdir(item_path) and item.startswith('pdfs-'):
                    directories.append(item)
            return sorted(directories)
        except Exception as e:
            logger.error(f"Error listing PDF directories: {str(e)}")
            return []

    def get_pdfs_in_directory(self, directory: str) -> List[str]:
        """Get all PDF files in a specific directory."""
        try:
            dir_path = os.path.join(self.base_directory, directory)
            if not os.path.exists(dir_path):
                return []
            return [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
        except Exception as e:
            logger.error(f"Error listing PDFs in directory {directory}: {str(e)}")
            return []

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
        """Get currently loaded PDFs."""
        return self.selected_pdfs

    def load_selected_pdfs(self, directory: str, selected_pdfs: List[str]) -> str:
        """Load specific PDFs from a directory into the vector store."""
        try:
            self.current_directory = directory
            self.selected_pdfs = selected_pdfs
            
            dir_path = os.path.join(self.base_directory, directory)
            
            documents = []
            loaded_count = 0
            
            for pdf_name in selected_pdfs:
                file_path = os.path.join(dir_path, pdf_name)
                if os.path.exists(file_path):
                    text = self.read_pdf(file_path)
                    if text:
                        logger.info(f"Successfully loaded PDF: {pdf_name}")
                        doc = Document(
                            page_content=text,
                            metadata={"source": pdf_name, "directory": directory}
                        )
                        documents.append(doc)
                        loaded_count += 1
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=400,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                )
                texts = text_splitter.split_documents(documents)
                
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
                logger.info(f"✅ Vector store initialized with {len(texts)} text chunks from {loaded_count} PDFs")
                return f"✅ Successfully loaded {loaded_count} PDFs with {len(texts)} text chunks from {directory}"
            else:
                return "❌ No valid PDFs could be loaded"
        
        except Exception as e:
            logger.error(f"Error loading selected PDFs: {str(e)}")
            return f"❌ Error loading PDFs: {str(e)}"

    def initialize_vector_store(self) -> None:
        """Legacy method for backward compatibility - loads from ./pdfs directory."""
        try:
            pdf_directory = os.path.join(self.base_directory, "pdfs")
            if not os.path.exists(pdf_directory):
                os.makedirs(pdf_directory)
                logger.info(f"Created PDF directory: {pdf_directory}")

            pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
            
            if not pdf_files:
                logger.warning("⚠️ No PDF files found in ./pdfs directory.")
                print("⚠️ No PDF files found in ./pdfs directory.")
                return

            # Load all PDFs from the pdfs directory
            result = self.load_selected_pdfs("pdfs", pdf_files)
            print(result)
        
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            print(f"❌ Error initializing vector store: {str(e)}")
            raise

    def query_documents(self, query: str, k: int = 5) -> str:
        try:
            if not self.selected_pdfs:
                return "No PDF documents have been loaded yet. Use /select to choose PDFs."

            if self.vector_store is None:
                return "Vector store not initialized."
            
            relevant_docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            if not relevant_docs_with_scores:
                return "No relevant information found."

            formatted_responses = []
            for doc, score in relevant_docs_with_scores:
                if score < 0.6:
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
        """Reload currently selected PDFs."""
        try:
            if not self.current_directory or not self.selected_pdfs:
                return "No PDFs currently selected. Use /select to choose PDFs first."
            
            return self.load_selected_pdfs(self.current_directory, self.selected_pdfs)
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
        if len(self.conversations[user_id]) > 20:
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
        # Store user selection states
        self.user_selections: Dict[int, Dict[str, Any]] = {}

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
        
        pdf_status = f"📚 PDFs Loaded: {len(self.doc_store.selected_pdfs)}" if self.doc_store.selected_pdfs else "⚠️ No PDFs loaded"
        
        welcome_message = f"""
👋 Hello! You can ask me anything about the loaded PDFs.

{pdf_status}
Messages remaining: {remaining_messages}

Commands:
/start - Show this message
/select - Choose PDF directories and files
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
/select - Choose PDF directories and files
/clear - Clear conversation history
/quota - Check remaining messages
/pdfs - List loaded PDFs
/reload - Reload PDFs
        """
        await update.message.reply_text(help_text)

    async def select_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /select command - show available PDF directories."""
        user_id = update.effective_user.id
        directories = self.doc_store.get_pdf_directories()
        
        if not directories:
            await update.message.reply_text("❌ No PDF directories found. Please create directories with names starting with 'pdfs-' (e.g., pdfs-analitica, pdfs-sistemas1)")
            return
        
        # Create inline keyboard with directory options
        keyboard = []
        for directory in directories:
            keyboard.append([InlineKeyboardButton(f"📁 {directory}", callback_data=f"dir:{directory}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📁 Select a PDF directory:",
            reply_markup=reply_markup
        )

    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        data = query.data
        
        if data.startswith("dir:"):
            # Directory selection
            directory = data[4:]  # Remove "dir:" prefix
            await self.show_pdf_selection(query, directory)
            
        elif data.startswith("pdf:"):
            # PDF selection toggle
            parts = data[4:].split(":", 1)  # Remove "pdf:" prefix and split once
            if len(parts) == 2:
                directory, pdf_name = parts
                await self.toggle_pdf_selection(query, directory, pdf_name)
        
        elif data.startswith("load:"):
            # Load selected PDFs
            directory = data[5:]  # Remove "load:" prefix
            await self.load_selected_pdfs(query, directory)
        
        elif data.startswith("back:"):
            # Go back to directory selection
            await self.show_directory_selection(query)

    async def show_directory_selection(self, query) -> None:
        """Show directory selection interface."""
        directories = self.doc_store.get_pdf_directories()
        
        keyboard = []
        for directory in directories:
            keyboard.append([InlineKeyboardButton(f"📁 {directory}", callback_data=f"dir:{directory}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📁 Select a PDF directory:",
            reply_markup=reply_markup
        )

    async def show_pdf_selection(self, query, directory: str) -> None:
        """Show PDF selection interface for a specific directory."""
        user_id = query.from_user.id
        pdfs = self.doc_store.get_pdfs_in_directory(directory)
        
        if not pdfs:
            await query.edit_message_text(
                f"❌ No PDF files found in {directory}",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back:directories")]])
            )
            return
        
        # Initialize user selection if not exists
        if user_id not in self.user_selections:
            self.user_selections[user_id] = {}
        
        if directory not in self.user_selections[user_id]:
            self.user_selections[user_id][directory] = set()
        
        # Create keyboard with PDF options
        keyboard = []
        selected_pdfs = self.user_selections[user_id][directory]
        
        for pdf in pdfs:
            status = "✅" if pdf in selected_pdfs else "⬜"
            keyboard.append([InlineKeyboardButton(f"{status} {pdf}", callback_data=f"pdf:{directory}:{pdf}")])
        
        # Add control buttons
        keyboard.append([
            InlineKeyboardButton("🔙 Back", callback_data="back:directories"),
            InlineKeyboardButton("📥 Load Selected", callback_data=f"load:{directory}")
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        selected_count = len(selected_pdfs)
        total_count = len(pdfs)
        
        await query.edit_message_text(
            f"📄 Select PDFs from {directory}:\n\nSelected: {selected_count}/{total_count}",
            reply_markup=reply_markup
        )

    async def toggle_pdf_selection(self, query, directory: str, pdf_name: str) -> None:
        """Toggle PDF selection and refresh the interface."""
        user_id = query.from_user.id
        
        # Initialize if needed
        if user_id not in self.user_selections:
            self.user_selections[user_id] = {}
        if directory not in self.user_selections[user_id]:
            self.user_selections[user_id][directory] = set()
        
        # Toggle selection
        selected_pdfs = self.user_selections[user_id][directory]
        if pdf_name in selected_pdfs:
            selected_pdfs.remove(pdf_name)
        else:
            selected_pdfs.add(pdf_name)
        
        # Refresh the interface
        await self.show_pdf_selection(query, directory)

    async def load_selected_pdfs(self, query, directory: str) -> None:
        """Load the selected PDFs into the vector store."""
        user_id = query.from_user.id
        
        if user_id not in self.user_selections or directory not in self.user_selections[user_id]:
            await query.edit_message_text("❌ No PDFs selected. Please select PDFs first.")
            return
        
        selected_pdfs = list(self.user_selections[user_id][directory])
        
        if not selected_pdfs:
            await query.edit_message_text("❌ No PDFs selected. Please select at least one PDF.")
            return
        
        # Show loading message
        await query.edit_message_text("📥 Loading selected PDFs... Please wait.")
        
        # Load the PDFs
        result = self.doc_store.load_selected_pdfs(directory, selected_pdfs)
        
        # Show result
        keyboard = [[InlineKeyboardButton("🔙 Back to Selection", callback_data="back:directories")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"{result}\n\nLoaded PDFs:\n" + "\n".join(f"📄 {pdf}" for pdf in selected_pdfs),
            reply_markup=reply_markup
        )

    async def list_pdfs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        pdfs = self.doc_store.get_loaded_pdfs()
        if pdfs:
            current_dir = self.doc_store.current_directory or "Unknown"
            pdf_list = "\n".join(f"📄 {pdf}" for pdf in pdfs)
            await update.message.reply_text(f"Loaded PDFs from {current_dir}:\n\n{pdf_list}")
        else:
            await update.message.reply_text("No PDFs currently loaded. Use /select to choose PDFs.")

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

            relevant_context = self.doc_store.query_documents(user_message, k=5)
            
            if relevant_context:
                system_message = f"""You are a helpful assistant with access to document knowledge.
Answer based on this information from the documents:

{relevant_context}

Cite sources when using document information. Provide comprehensive, detailed responses
as your answers will be sent as PDF files with no length limitations."""
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

            # Add special instruction for filename generation
            filename_instruction = """
After providing your complete answer, on a new line add: 
[FILENAME: a_descriptive_filename_based_on_content]

The filename should be descriptive of the content, use underscores instead of spaces, 
end with .pdf, and be between 5-50 characters long. This will be used as the actual 
filename when saving the document.
            """
            
            modified_user_message = f"{user_message}\n\n{filename_instruction}"
            messages.append({"role": "user", "content": modified_user_message})

            # Call Claude API
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10000,
                temperature=0.1,
                messages=messages
            )

            ai_response = response.content[0].text
            
            # Extract the filename if provided
            import re
            filename_match = re.search(r'\[FILENAME:\s*([^\]]+)\]', ai_response)
            
            if filename_match:
                suggested_filename = filename_match.group(1).strip()
                # Remove the filename line from the response
                ai_response = re.sub(r'\[FILENAME:\s*[^\]]+\]', '', ai_response).strip()
                
                # Ensure filename ends with .pdf and has valid characters
                suggested_filename = suggested_filename.replace(' ', '_')
                if not suggested_filename.lower().endswith('.pdf'):
                    suggested_filename += '.pdf'
                    
                # Sanitize filename to avoid any problematic characters
                import string
                valid_chars = f"-_.{string.ascii_letters}{string.digits}"
                suggested_filename = ''.join(c for c in suggested_filename if c in valid_chars)
                
                # Limit length and ensure it's not empty
                if len(suggested_filename) > 50:
                    suggested_filename = suggested_filename[:46] + '.pdf'
                if len(suggested_filename) < 5:
                    suggested_filename = f"Claude_Response_{datetime.now().strftime('%Y%m%d')}.pdf"
            else:
                # Fallback filename if none provided
                suggested_filename = f"Claude_Response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            # Store the cleaned response in conversation history
            self.rate_limiter.add_message(user_id)
            self.conversation_manager.add_message(user_id, "user", user_message)
            self.conversation_manager.add_message(user_id, "assistant", ai_response)

            remaining_messages = self.rate_limiter.get_remaining_messages(user_id)
            
            # Send a brief preview message first
            preview_length = min(150, len(ai_response))
            preview = ai_response[:preview_length]
            if len(ai_response) > preview_length:
                preview += "..."
            
            await update.message.reply_text(
                f"📝 Response preview:\n\n{preview}\n\nFull response in attached PDF: {suggested_filename}"
            )
            
            # Create a PDF from the response
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_JUSTIFY
            from io import BytesIO
            import textwrap
            from datetime import datetime
            
            # Create a PDF buffer
            buffer = BytesIO()
            
            # Set up the PDF document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72, leftMargin=72,
                topMargin=72, bottomMargin=72
            )
            
            # Create styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='Justify',
                alignment=TA_JUSTIFY,
                fontSize=11,
                leading=14
            ))
            
            # Create content
            content = []
            
            # Add title - use a cleaned version of the filename as title
            title_style = styles['Heading1']
            title_text = suggested_filename.replace('_', ' ').replace('.pdf', '')
            content.append(Paragraph(title_text, title_style))
            content.append(Spacer(1, 12))
            
            # Add query
            query_style = styles['Heading2']
            content.append(Paragraph("Query:", query_style))
            content.append(Paragraph(user_message, styles['Normal']))
            content.append(Spacer(1, 12))
            
            # Add response
            response_style = styles['Heading2']
            content.append(Paragraph("Response:", response_style))
            
            # Break the response into paragraphs and add each one
            paragraphs = ai_response.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Handle markdown headers
                    if para.startswith('# '):
                        content.append(Paragraph(para[2:], styles['Heading1']))
                    elif para.startswith('## '):
                        content.append(Paragraph(para[3:], styles['Heading2']))
                    elif para.startswith('### '):
                        content.append(Paragraph(para[4:], styles['Heading3']))
                    # Handle code blocks with simple formatting
                    elif para.strip().startswith('```') and para.strip().endswith('```'):
                        code_text = para.strip()[3:-3]
                        code_lines = code_text.split('\n')
                        if code_lines and not code_lines[0].strip():
                            code_lines = code_lines[1:]
                        code_para = '<font face="Courier">' + '<br/>'.join(code_lines) + '</font>'
                        content.append(Paragraph(code_para, styles['Normal']))
                    else:
                        # Regular paragraph
                        content.append(Paragraph(para, styles['Justify']))
                    content.append(Spacer(1, 10))
            
            # Add footer with timestamp
            footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            content.append(Spacer(1, 20))
            content.append(Paragraph(footer_text, styles['Normal']))
            
            # Build the PDF
            doc.build(content)
            
            # Reset buffer position
            buffer.seek(0)
            
            # Send the PDF file
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=buffer,
                filename=suggested_filename,
                caption=f"📊 {remaining_messages} messages remaining today."
            )
            logger.info(f"Response sent as PDF '{suggested_filename}' to user {user_id}, {remaining_messages} messages remaining")

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
            self.application.add_handler(CommandHandler("select", self.select_command))
            self.application.add_handler(CommandHandler("clear", self.clear_history))
            self.application.add_handler(CommandHandler("quota", self.check_quota))
            self.application.add_handler(CommandHandler("pdfs", self.list_pdfs))
            self.application.add_handler(CommandHandler("reload", self.reload_pdfs))
            self.application.add_handler(CallbackQueryHandler(self.handle_callback_query))
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