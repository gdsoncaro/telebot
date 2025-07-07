import os
import telebot
import replicate
from anthropic import Anthropic
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# Initialize clients
bot = telebot.TeleBot(os.getenv('RECRAFT_TELEGRAM_BOT_TOKEN'))
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

# Define suitable styles for comforting images
COMFORTING_STYLES = [
    "digital_illustration/2d_art_poster",  # Soft, artistic style
    "digital_illustration/hand_drawn",     # Gentle, personal feel
    "realistic_image/natural_light",       # Warm, natural look
    "digital_illustration/grain",          # Soft, textured feel
    "realistic_image/studio_portrait"      # Clean, professional look
]

def generate_supportive_response(user_message):
    """Generate an empathetic and supportive response using Claude"""
    try:
        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": f"""As a compassionate companion, provide a warm and supportive response to this message: "{user_message}"
                Keep the response concise (2-3 sentences) but genuine and empathetic. Focus on validation, understanding, and gentle encouragement."""
            }]
        )
        return str(message.content)
    except Exception as e:
        raise Exception(f"Response generation failed: {str(e)}")

def generate_comforting_image_prompt(emotional_context):
    """Generate a prompt for a calming, supportive image"""
    try:
        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0.6,
            messages=[{
                "role": "user",
                "content": f"""Create a soothing and uplifting image prompt based on this emotional context: "{emotional_context}"
                Focus on calming elements like nature, soft colors, peaceful scenes, or comforting symbols.
                The image should evoke feelings of peace, hope, and support. Keep it under 75 words."""
            }]
        )
        return str(message.content).strip()
    except Exception as e:
        raise Exception(f"Prompt generation failed: {str(e)}")

def generate_image(prompt):
    """Generate a calming image using Recraft API"""
    try:
        # Choose a random appropriate style
        chosen_style = random.choice(COMFORTING_STYLES)
        
        output = replicate.run(
            "recraft-ai/recraft-v3",
            input={
                "size": "1365x1024",
                "style": chosen_style,
                "prompt": f"{prompt}, peaceful atmosphere, calming scene"
            }
        )
        if not output:
            raise Exception("No image was generated")
        return output[0] if isinstance(output, list) and len(output) > 0 else output
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

def safe_reply(bot, chat_id, text, reply_to_message_id=None):
    """Safely send a message with error handling"""
    try:
        return bot.send_message(
            chat_id,
            text,
            reply_to_message_id=reply_to_message_id,
            parse_mode='Markdown'
        )
    except Exception as e:
        # Fallback to plain text if markdown fails
        return bot.send_message(
            chat_id,
            text,
            reply_to_message_id=reply_to_message_id,
            parse_mode=None
        )

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = (
        "Hello! 🌸 I'm here to listen and support you.\n\n"
        "Feel free to share whatever's on your mind - your thoughts, feelings, or anything you'd like to talk about.\n"
        "I'll respond with understanding and create a peaceful image to help bring you comfort.\n\n"
        "You can start our conversation anytime. 🤗"
    )
    safe_reply(bot, message.chat.id, welcome_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        if not message.text:
            safe_reply(bot, message.chat.id, "Please share your thoughts with me - I'm here to listen. 🌱")
            return
        
        # Send gentle processing message
        processing_msg = safe_reply(bot, message.chat.id, "Taking a moment to reflect... 🌸")
        
        try:
            # Generate supportive response
            response = generate_supportive_response(message.text)
            if response:
                safe_reply(bot, message.chat.id, f"{response}")
            
            # Generate and send comforting image
            image_prompt = generate_comforting_image_prompt(message.text)
            if image_prompt:
                image_url = generate_image(image_prompt)
                if image_url:
                    bot.send_photo(
                        message.chat.id,
                        image_url,
                        caption="I created this peaceful image for you... 🌸✨"
                    )
        except Exception as inner_e:
            print(f"Error during response generation: {str(inner_e)}")
            safe_reply(
                bot,
                message.chat.id,
                "I'm still here with you, but I'm having a moment of difficulty expressing myself. Please feel free to continue sharing. 🌱"
            )
        
        # Delete the processing message if it exists
        if processing_msg:
            try:
                bot.delete_message(message.chat.id, processing_msg.message_id)
            except:
                pass
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        try:
            safe_reply(
                bot,
                message.chat.id,
                "I apologize for the interruption in our conversation. I'm here to listen if you'd like to share again. 🌱"
            )
        except:
            pass

def main():
    print("Companion Bot is here to support...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()