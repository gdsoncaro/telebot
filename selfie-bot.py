import os
import telebot
import replicate
from anthropic import Anthropic
from dotenv import load_dotenv
import base64
import tempfile

# Load environment variables
load_dotenv()

# Initialize clients
bot = telebot.TeleBot(os.getenv('RECRAFT_TELEGRAM_BOT_TOKEN'))
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

# Store user states (photo paths and whether they're waiting for prompts)
user_states = {}

def download_photo(file_info):
    """Download photo from Telegram and save to a temporary file"""
    file_path = bot.get_file(file_info.file_id).file_path
    downloaded_file = bot.download_file(file_path)
    
    # Create temporary file with .jpg extension
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_file.write(downloaded_file)
    temp_file.close()
    
    return temp_file.name

def analyze_image(photo_path):
    """Analyze image using Claude Vision API"""
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
                        "text": "Analyze this selfie. Describe the person's appearance, pose, and lighting conditions."
                    }
                ]
            }]
        )
        return message.content
    except Exception as e:
        raise Exception(f"Image analysis failed: {str(e)}")

def modify_image(image_path, prompt, analysis):
    """Modify the image using Recraft V3 API with improved preservation of original features"""
    try:
        # Extract key features from Claude's analysis
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
                "num_samples": 1,  # Generate only one image
                "num_inference_steps": 50,  # Increase steps for better quality
                "guidance_scale": 7.5  # Adjust guidance scale for better prompting
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
        "2. Tell me how you'd like to modify it\n"
        "3. I'll analyze and create a modified version based on your prompt\n\n"
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
        
        # Analyze the image using Claude Vision
        bot.reply_to(message, "Analyzing your selfie... 🔍")
        analysis = analyze_image(photo_path)
        
        # Store the photo path, analysis, and set state to waiting for prompt
        user_states[message.from_user.id] = {
            'photo_path': photo_path,
            'analysis': analysis,
            'waiting_for_prompt': True
        }
        
        bot.reply_to(message, 
            "Great! Now tell me how you'd like to modify your photo.\n\n"
            "For example:\n"
            "• 'Make it look like a professional headshot'\n"
            "• 'Add a cyberpunk style'\n"
            "• 'Transform it into an oil painting'"
        )
        
    except Exception as e:
        bot.reply_to(message, "Sorry, I had trouble processing your photo. Please try again with a clear selfie.")
        print(f"Error: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_prompt(message):
    user_id = message.from_user.id
    
    # Check if we're waiting for a prompt from this user
    if user_id in user_states and user_states[user_id].get('waiting_for_prompt'):
        try:
            bot.reply_to(message, "Processing your request... This might take a moment ⏳")
            
            # Get the stored photo path and analysis
            photo_path = user_states[user_id]['photo_path']
            analysis = user_states[user_id]['analysis']
            
            # Modify the image based on the prompt and analysis
            modified_image_url = modify_image(photo_path, message.text, analysis)
            
            # Send the modified image
            bot.send_photo(
                message.chat.id,
                modified_image_url,
                caption="Here's your modified photo! ✨\nSend another selfie to start over."
            )
            
            # Clean up
            os.remove(photo_path)
            del user_states[user_id]
            
        except Exception as e:
            bot.reply_to(message, "Sorry, I had trouble modifying your photo. Please try again with a different prompt or photo.")
            print(f"Error: {str(e)}")
            
            # Clean up on error
            if user_id in user_states:
                if os.path.exists(user_states[user_id]['photo_path']):
                    os.remove(user_states[user_id]['photo_path'])
                del user_states[user_id]
    else:
        bot.reply_to(message, "Please send me a selfie first! 📸")

def main():
    print("Selfie Editor Bot is running...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()