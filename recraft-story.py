import os
import telebot
import replicate
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
bot = telebot.TeleBot(os.getenv('RECRAFT_TELEGRAM_BOT_TOKEN'))
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

def generate_story(prompt):
    """Generate a short story using Claude API based on user prompt"""
    try:
        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            messages=[{
                "role": "user",
                "content": f"Create a short, engaging story (2-3 paragraphs) based on this prompt: {prompt}. "
                          f"The story should be vivid and descriptive, suitable for generating an interesting image afterward."
            }]
        )
        return message.content
    except Exception as e:
        raise Exception(f"Story generation failed: {str(e)}")

def generate_image(prompt):
    """Generate an image using Recraft API based on the story"""
    try:
        # Ensure prompt is a string
        prompt_str = str(prompt)
        output = replicate.run(
            "recraft-ai/recraft-v3",
            input={
                "size": "1365x1024",
                "style": "any",
                "prompt": prompt_str
            }
        )
        # The API returns a list with one item - the image URL
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return output
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

def extract_image_prompt(story):
    """Extract a suitable image prompt from the story"""
    try:
        message = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0.5,
            messages=[{
                "role": "user",
                "content": f"Based on this story, create a detailed, visual prompt suitable for image generation. "
                          f"Focus on the main visual elements, atmosphere, and style. Keep it under 100 words:\n\n{story}"
            }]
        )
        # Ensure we return a string
        return str(message.content)
    except Exception as e:
        raise Exception(f"Prompt extraction failed: {str(e)}")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = (
        "Welcome to the Storytelling Bot! 🎭✨\n\n"
        "Share any prompt with me, and I'll:\n"
        "1. Create a unique story\n"
        "2. Generate an image based on the story\n\n"
        "Just type your prompt or use /imagine followed by your prompt!"
    )
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['imagine'])
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        # Extract prompt (either from /imagine command or direct message)
        prompt = message.text
        if prompt.startswith('/imagine'):
            prompt = prompt.replace('/imagine', '').strip()
        
        if not prompt:
            bot.reply_to(message, "Please provide a prompt for your story!")
            return
        
        # Send processing message
        processing_msg = bot.reply_to(message, "🎨 Creating your story and artwork...")
        
        # Generate story
        story = generate_story(prompt)
        bot.reply_to(message, f"Here's your story:\n\n{story}")
        
        # Extract image prompt from story
        image_prompt = extract_image_prompt(story)
        
        # Send a message indicating image generation is starting
        bot.reply_to(message, f"Now generating an image with this prompt:\n{image_prompt}")
        
        # Generate image
        image_url = generate_image(image_prompt)
        
        # Send the generated image with the prompt used
        bot.send_photo(
            message.chat.id,
            image_url,
            caption=f"🎨 Generated based on the story\nImage prompt: {image_prompt}"
        )
        
        # Delete the processing message
        bot.delete_message(message.chat.id, processing_msg.message_id)
        
    except Exception as e:
        error_message = f"Sorry, something went wrong: {str(e)}"
        bot.reply_to(message, error_message)

def main():
    print("Storytelling Bot is running...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()