import os
import telebot
import replicate
from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image
import io
import requests
import base64  # Added this import

# Load environment variables
load_dotenv()

# Initialize clients
bot = telebot.TeleBot(os.getenv('RECRAFT_TELEGRAM_BOT_TOKEN'))
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

# User wardrobe storage (in-memory for demo, consider using a database)
user_wardrobe = {}

def analyze_clothing_item(photo_path):
    """Analyze clothing item using Claude Vision"""
    try:
        with open(photo_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": "Analyze this clothing item. Describe its type, color, style, and potential occasions it would be suitable for. Format as JSON with keys: type, color, style, occasions"
                    }
                ]
            }]
        )
        return message.content
    except Exception as e:
        raise Exception(f"Clothing analysis failed: {str(e)}")

def generate_outfit_suggestion(wardrobe_items, occasion=None):
    """Generate outfit suggestions based on wardrobe items"""
    try:
        items_description = "\n".join([f"- {item}" for item in wardrobe_items])
        prompt = f"""Given these wardrobe items:
        {items_description}
        
        {"Create an outfit for " + occasion if occasion else "Create a stylish outfit combination"}. 
        Explain why these items work well together and suggest any accessories."""

        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return message.content
    except Exception as e:
        raise Exception(f"Outfit suggestion failed: {str(e)}")

def generate_outfit_image(outfit_description):
    """Generate a photorealistic image of the suggested outfit"""
    try:
        output = replicate.run(
            "recraft-ai/recraft-v3",
            input={
                "size": "1024x1024",
                "style": "realistic_image/studio_portrait",  # Changed to photorealistic style
                "prompt": f"Professional fashion photography of {outfit_description}. "
                         f"High-end fashion magazine style, studio lighting, clean white background, "
                         f"ultra-realistic, high detail, professional photography, 8k, high resolution"
            }
        )
        return output[0] if isinstance(output, list) and len(output) > 0 else output
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

def download_photo(file_info):
    """Download photo from Telegram"""
    file_path = bot.get_file(file_info.file_id).file_path
    downloaded_file = bot.download_file(file_path)
    file_name = f"temp_{file_info.file_id}.jpg"
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)
    return file_name

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = (
        "Welcome to your Personal Fashion Advisor! 👗✨\n\n"
        "I can help you with:\n"
        "• Analyzing your clothing items (send me a photo)\n"
        "• Suggesting outfit combinations (/outfit)\n"
        "• Creating outfit visualizations\n"
        "• Providing style advice (/advice)\n\n"
        "Let's start building your digital wardrobe! Send me a photo of any clothing item."
    )
    bot.reply_to(message, welcome_text)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        # Get the largest version of the photo
        file_info = bot.get_file(message.photo[-1].file_id)
        
        # Download and save photo temporarily
        photo_path = download_photo(file_info)
        
        # Analyze the clothing item
        bot.reply_to(message, "Analyzing your item... 👗")
        analysis = analyze_clothing_item(photo_path)
        
        # Store in user's wardrobe
        if message.from_user.id not in user_wardrobe:
            user_wardrobe[message.from_user.id] = []
        user_wardrobe[message.from_user.id].append(analysis)
        
        # Send analysis results
        bot.reply_to(message, 
            f"✨ Item Analysis ✨\n\n{analysis}\n\n"
            f"Item added to your wardrobe! You now have {len(user_wardrobe[message.from_user.id])} items."
        )
        
        # Clean up temporary file
        os.remove(photo_path)
        
    except Exception as e:
        bot.reply_to(message, "Sorry, I had trouble analyzing that item. Please try again with a clear photo.")
        print(f"Error: {str(e)}")

@bot.message_handler(commands=['outfit'])
def generate_outfit(message):
    try:
        user_id = message.from_user.id
        if user_id not in user_wardrobe or not user_wardrobe[user_id]:
            bot.reply_to(message, "Your wardrobe is empty! Send me some photos of your clothes first.")
            return
            
        bot.reply_to(message, "Creating a photorealistic outfit visualization... 📸")
        
        # Generate outfit suggestion
        suggestion = generate_outfit_suggestion(user_wardrobe[user_id])
        
        # Generate photorealistic outfit visualization
        image_url = generate_outfit_image(suggestion)
        
        # Send suggestion and image
        bot.reply_to(message, f"Here's your outfit suggestion:\n\n{suggestion}")
        if image_url:
            bot.send_photo(
                message.chat.id, 
                image_url, 
                caption="Here's how your outfit would look in a professional photo shoot ✨📸"
            )
            
    except Exception as e:
        bot.reply_to(message, "Sorry, I had trouble generating your outfit visualization. Please try again.")
        print(f"Error: {str(e)}")

@bot.message_handler(commands=['advice'])
def get_style_advice(message):
    try:
        bot.reply_to(message, "What occasion are you dressing for? (e.g., work, casual, party, formal)")
        bot.register_next_step_handler(message, provide_style_advice)
    except Exception as e:
        bot.reply_to(message, "Sorry, I had trouble processing your request. Please try again.")
        print(f"Error: {str(e)}")

def provide_style_advice(message):
    try:
        occasion = message.text.lower()
        user_id = message.from_user.id
        
        if user_id in user_wardrobe and user_wardrobe[user_id]:
            suggestion = generate_outfit_suggestion(user_wardrobe[user_id], occasion)
            bot.reply_to(message, f"Style advice for {occasion}:\n\n{suggestion}")
        else:
            bot.reply_to(message, 
                f"I can give better advice if you share some of your clothes with me first!\n\n"
                f"Here's some general advice for now:\n"
                f"For {occasion}, consider wearing elegant yet comfortable pieces that match the setting..."
            )
    except Exception as e:
        bot.reply_to(message, "Sorry, I had trouble generating style advice. Please try again.")
        print(f"Error: {str(e)}")

def main():
    print("Fashion Advisor Bot is running...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()