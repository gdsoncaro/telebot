import os
import telebot
from telebot.handler_backends import State, StatesGroup
from telebot.storage import StateMemoryStorage
from telebot import types
import replicate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize bot with state storage
state_storage = StateMemoryStorage()
bot = telebot.TeleBot(os.getenv('RECRAFT_TELEGRAM_BOT_TOKEN'), state_storage=state_storage)

# Initialize Replicate client
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

# Store user prompts temporarily
user_prompts = {}

# Complete style options with friendly names
STYLES = {
    "Automatic (Any Style)": "any",
    "Realistic Photo": "realistic_image",
    "Digital Illustration": "digital_illustration",
    "Pixel Art": "digital_illustration/pixel_art",
    "Hand Drawn": "digital_illustration/hand_drawn",
    "Grainy Digital": "digital_illustration/grain",
    "Infantile Sketch": "digital_illustration/infantile_sketch",
    "2D Art Poster": "digital_illustration/2d_art_poster",
    "3D Handmade": "digital_illustration/handmade_3d",
    "Hand Drawn Outline": "digital_illustration/hand_drawn_outline",
    "Color Engraving": "digital_illustration/engraving_color",
    "2D Art Poster Alt": "digital_illustration/2d_art_poster_2",
    "Black & White": "realistic_image/b_and_w",
    "Hard Flash Photo": "realistic_image/hard_flash",
    "HDR Photo": "realistic_image/hdr",
    "Natural Light": "realistic_image/natural_light",
    "Studio Portrait": "realistic_image/studio_portrait",
    "Enterprise": "realistic_image/enterprise",
    "Motion Blur": "realistic_image/motion_blur"
}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! Use /imagine followed by your prompt to generate an image, or /styles to see available styles.")

@bot.message_handler(commands=['styles'])
def show_styles(message):
    styles_list = "Available styles:\n\n" + "\n".join(f"• {style}" for style in STYLES.keys())
    bot.reply_to(message, styles_list)

@bot.message_handler(commands=['imagine'])
def start_image_generation(message):
    prompt = message.text.replace('/imagine', '').strip()
    
    if not prompt:
        bot.reply_to(message, "Please provide a prompt after /imagine")
        return
    
    # Store the prompt
    user_prompts[message.chat.id] = prompt
    
    # Create style selection keyboard with pagination
    markup = types.InlineKeyboardMarkup(row_width=2)
    
    # Create buttons in pairs
    buttons = []
    for style_name, style_value in STYLES.items():
        buttons.append(types.InlineKeyboardButton(
            text=style_name, 
            callback_data=f"style_{style_value}"
        ))
    
    # Add buttons in pairs to the markup
    for i in range(0, len(buttons), 2):
        if i + 1 < len(buttons):
            markup.row(buttons[i], buttons[i + 1])
        else:
            markup.row(buttons[i])
    
    bot.reply_to(message, "Choose a style for your image:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith('style_'))
def handle_style_selection(call):
    try:
        # Extract style and prompt
        selected_style = call.data.replace('style_', '')
        prompt = user_prompts.get(call.message.chat.id)
        
        if not prompt:
            bot.answer_callback_query(call.id, "Sorry, your prompt was lost. Please try /imagine again.")
            return
        
        # Send processing message
        processing_msg = bot.send_message(call.message.chat.id, "🎨 Generating your image...")
        
        # Delete the style selection message
        bot.delete_message(call.message.chat.id, call.message.message_id)
        
        # Generate image
        output = replicate.run(
            "recraft-ai/recraft-v3",
            input={
                "size": "1365x1024",
                "style": selected_style,
                "prompt": prompt
            }
        )
        
        # Find the friendly name of the style
        style_name = next((name for name, value in STYLES.items() if value == selected_style), selected_style)
        
        # Send the generated image
        bot.send_photo(
            call.message.chat.id, 
            output, 
            caption=f"Prompt: {prompt}\nStyle: {style_name}"
        )
        
        # Delete the processing message
        bot.delete_message(call.message.chat.id, processing_msg.message_id)
        
        # Clean up stored prompt
        del user_prompts[call.message.chat.id]
        
    except Exception as e:
        bot.send_message(call.message.chat.id, f"Sorry, an error occurred: {str(e)}")
    
    finally:
        bot.answer_callback_query(call.id)

def main():
    print("Bot is running...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()