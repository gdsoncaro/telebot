import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import asyncio
import nest_asyncio

# Apply nest_asyncio to fix Colab's event loop issue
nest_asyncio.apply()

# Configure your tokens
TELEGRAM_TOKEN = "7668067210:AAER7jlgypqGs2LtpVdKo9lwUFXel7r0ujg"  # Replace with your Telegram token
MODEL_ID = "EleutherAI/gpt-neo-2.7B"

class LLMBot:
    def __init__(self):
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=quantization_config
        )
        print("Model loaded successfully!")

    async def generate_response(self, prompt: str) -> str:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        return response

class TelegramBot:
    def __init__(self):
        self.llm_bot = LLMBot()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Hello! I'm your AI assistant powered by Mistral. How can I help you today?"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_message = update.message.text
            print(f"Received message: {user_message}")
            response = await self.llm_bot.generate_response(user_message)
            print(f"Generated response: {response}")
            await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text(
                "I encountered an error processing your request. Please try again."
            )
            print(f"Error: {str(e)}")

    async def run(self):
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        print("Starting bot...")
        await application.run_polling()

# Run the bot
bot = TelegramBot()
asyncio.run(bot.run())