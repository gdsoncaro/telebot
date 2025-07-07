import os
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import replicate
from anthropic import Anthropic
from dotenv import load_dotenv
import base64
import tempfile
import json

# Load environment variables
load_dotenv()

# Initialize clients
bot = telebot.TeleBot(os.getenv('RECRAFT_TELEGRAM_BOT_TOKEN'))
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

# Store user states
user_states = {}

def download_photo(file_info):
    """Download photo from Telegram and save to a temporary file"""
    file_path = bot.get_file(file_info.file_id).file_path
    downloaded_file = bot.download_file(file_path)
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_file.write(downloaded_file)
    temp_file.close()
    
    return temp_file.name

def analyze_image_and_suggest_styles(photo_path):
    """Analyze image using Claude Vision and get style suggestions"""
    try:
        with open(photo_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
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
                        "text": """Analyze this selfie and return ONLY a JSON object with two keys:
                        {
                            "analysis": "brief description of the image",
                            "suggestions": [
                                {
                                    "label": "short button text",
                                    "prompt": "detailed modification instructions"
                                },
                                ... (total of 6 suggestions)
                            ]
                        }
                        Make suggestions specific to what you see in the image."""
                    }
                ]
            }]
        )
        
        # Extract text from Claude's response
        response_text = message.content[0].text if isinstance(message.content, list) else message.content
        
        # Parse the JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # If no code blocks, try to find any JSON-like structure
                json_match = re.search(r'{[\s\S]*?}', response_text)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    raise Exception("Could not find valid JSON in response")
                    
    except Exception as e:
        print(f"Full error in analyze_image_and_suggest_styles: {str(e)}")
        print(f"Claude response: {message.content}")
        raise Exception(f"Image analysis failed: {str(e)}")

def create_style_keyboard(suggestions):
    """Create inline keyboard from style suggestions"""
    keyboard = InlineKeyboardMarkup(row_width=2)
    buttons = []
    
    for suggestion in suggestions:
        # Store the full prompt as callback data
        callback_data = f"style_{suggestions.index(suggestion)}"  # Using index to keep callback data small
        buttons.append(InlineKeyboardButton(text=suggestion['label'], callback_data=callback_data))
    
    keyboard.add(*buttons)
    return keyboard

def modify_image(image_path, prompt, analysis):
    """Modify the image using Recraft V3 API"""
    try:
        enhanced_prompt = f"""Modify this exact person's portrait while strictly preserving their facial features, 
        identity, and key characteristics: {analysis}. 
        
        Apply the following style changes: {prompt}.
        
        Critical requirements:
        - Maintain exact facial structure, features, and proportions
        - Keep the same identity clearly recognizable
        - Only modify style elements as requested"""
        
        output = replicate.run(
            "recraft-ai/recraft-v3",
            input={
                "size": "1024x1024",
                "style": "realistic_image/studio_portrait",
                "prompt": f"Ultra-realistic portrait modification of the exact same person. {enhanced_prompt}. "
                         f"Maintain perfect likeness, same facial features, same identity. "
                         f"Professional photography, studio lighting, high detail, 8k resolution. "
                         f"This must look like the exact same person with minimal style changes.",
                "image": open(image_path, "rb"),
                "num_samples": 1,
                "num_inference_steps": 50,
                "guidance_scale": 7.5
            }
        )
        return output[0] if isinstance(output, list) and len(output) > 0 else output
    except Exception as e:
        raise Exception(f"Image modification failed: {str(e)}")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = (
        "Welcome to the Selfie Editor Bot! 📸✨\n\n"
        "How to use:\n"
        "1. Send me a selfie\n"
        "2. Choose from suggested style modifications\n"
        "3. I'll create a modified version based on your choice\n\n"
        "Send me your selfie to get started!"
    )
    bot.reply_to(message, welcome_text)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        # Get the largest version of the photo
        file_info = bot.get_file(message.photo[-1].file_id)
        
        # Download and save photo temporarily
        photo_path = download_photo(file_info)
        
        # Analyze the image and get style suggestions
        bot.reply_to(message, "Analyzing your selfie and generating style suggestions... 🔍")
        analysis_result = analyze_image_and_suggest_styles(photo_path)
        
        # Store the photo path, analysis, and suggestions
        user_states[message.from_user.id] = {
            'photo_path': photo_path,
            'analysis': analysis_result['analysis'],
            'suggestions': analysis_result['suggestions']
        }
        
        # Create and send inline keyboard with style options
        keyboard = create_style_keyboard(analysis_result['suggestions'])
        bot.reply_to(
            message,
            "Choose a style modification:",
            reply_markup=keyboard
        )
        
    except Exception as e:
        bot.reply_to(message, "Sorry, I had trouble processing your photo. Please try again with a clear selfie.")
        print(f"Error: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('style_'))
def handle_style_selection(call):
    try:
        user_id = call.from_user.id
        if user_id in user_states:
            # Get selected style index
            style_index = int(call.data.split('_')[1])
            selected_style = user_states[user_id]['suggestions'][style_index]
            
            # Update message to show selection
            bot.edit_message_reply_markup(
                call.message.chat.id,
                call.message.message_id,
                reply_markup=None
            )
            bot.answer_callback_query(call.id)
            
            # Send processing message
            bot.send_message(call.message.chat.id, "Processing your request... ⏳")
            
            # Modify image
            photo_path = user_states[user_id]['photo_path']
            analysis = user_states[user_id]['analysis']
            modified_image_url = modify_image(photo_path, selected_style['prompt'], analysis)
            
            # Send modified image
            bot.send_photo(
                call.message.chat.id,
                modified_image_url,
                caption=f"Here's your photo in {selected_style['label']} style! ✨\nSend another selfie to try more styles."
            )
            
            # Clean up
            os.remove(photo_path)
            del user_states[user_id]
            
        else:
            bot.answer_callback_query(call.id, "Session expired. Please send a new photo.")
            
    except Exception as e:
        bot.answer_callback_query(call.id, "Sorry, something went wrong. Please try again.")
        print(f"Error: {str(e)}")
        
        # Clean up on error
        if user_id in user_states:
            if os.path.exists(user_states[user_id]['photo_path']):
                os.remove(user_states[user_id]['photo_path'])
            del user_states[user_id]

def main():
    print("Selfie Editor Bot is running...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()