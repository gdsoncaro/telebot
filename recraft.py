import os
import telebot
import replicate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Telegram bot
bot = telebot.TeleBot(os.getenv('RECRAFT_TELEGRAM_BOT_TOKEN'))

# Initialize Replicate client
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Use /imagine followed by your prompt to generate an image.")

@bot.message_handler(commands=['imagine'])
def generate_image(message):
    try:
        # Extract the prompt (everything after /imagine)
        prompt = message.text.replace('/imagine', '').strip()
        
        if not prompt:
            bot.reply_to(message, "Please provide a prompt after /imagine")
            return
        
        # Send "generating" message
        processing_msg = bot.reply_to(message, "🎨 Generating your image...")
        
        # Generate image using Replicate API
        output = replicate.run(
            "recraft-ai/recraft-v3",
            input={
                "size": "1365x1024",
                "style": "any",
                "prompt": prompt
            }
        )
        
        # Send the generated image
        bot.send_photo(message.chat.id, output, caption=f"Prompt: {prompt}")
        
        # Delete the "generating" message
        bot.delete_message(message.chat.id, processing_msg.message_id)
        
    except Exception as e:
        bot.reply_to(message, f"Sorry, an error occurred: {str(e)}")

def main():
    print("Bot is running...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()