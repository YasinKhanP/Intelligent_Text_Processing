import tkinter as tk
from tkinter import messagebox, ttk
from transformers import T5ForConditionalGeneration, T5Tokenizer
from gtts import gTTS
import openai
from PIL import Image
from io import BytesIO
import requests
import pygame
import tempfile
import threading
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize OpenAI client
from openai import OpenAI

client = OpenAI(api_key=openai_api_key)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
try:
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model or tokenizer: {str(e)}")
    raise

# Initialize pygame mixer
pygame.mixer.init()

# Create the main application window
root = tk.Tk()
root.title("Text Processing Application")


# Function to generate summary
def generate_summary():
    try:
        input_text = input_text_area.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Error", "Please enter some input text.")
            return

        # Disable buttons and show progress
        set_buttons_state(tk.DISABLED)
        progress_bar.start()

        # Run model inference in a separate thread
        def run_inference():
            try:
                input_encoding = tokenizer(
                    input_text,
                    max_length=1024,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)
                summary_ids = model.generate(
                    input_encoding['input_ids'],
                    num_beams=4,
                    max_length=128,
                    early_stopping=True
                )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Update GUI in main thread
                root.after(0, lambda: (
                    output_text_area.delete("1.0", tk.END),
                    output_text_area.insert(tk.END, summary),
                    messagebox.showinfo("Success", "Summary Generated!"),
                    progress_bar.stop(),
                    set_buttons_state(tk.NORMAL)
                ))
            except Exception as e:
                root.after(0, lambda: (
                    messagebox.showerror("Error", f"Summary generation failed: {str(e)}"),
                    progress_bar.stop(),
                    set_buttons_state(tk.NORMAL)
                ))

        threading.Thread(target=run_inference, daemon=True).start()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        progress_bar.stop()
        set_buttons_state(tk.NORMAL)


# Function to convert summary to speech
def text_to_speech():
    try:
        summary_text = output_text_area.get("1.0", tk.END).strip()
        if not summary_text:
            messagebox.showerror("Error", "No summary to convert!")
            return

        set_buttons_state(tk.DISABLED)
        progress_bar.start()

        def run_tts():
            try:
                tts = gTTS(text=summary_text, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_file:
                    tts.save(temp_file.name)
                    pygame.mixer.music.load(temp_file.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)

                root.after(0, lambda: (
                    messagebox.showinfo("Success", "Speech played successfully!"),
                    progress_bar.stop(),
                    set_buttons_state(tk.NORMAL)
                ))
            except Exception as e:
                root.after(0, lambda: (
                    messagebox.showerror("Error", f"TTS failed: {str(e)}"),
                    progress_bar.stop(),
                    set_buttons_state(tk.NORMAL)
                ))

        threading.Thread(target=run_tts, daemon=True).start()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        progress_bar.stop()
        set_buttons_state(tk.NORMAL)


# Function to generate image using DALL-E
def generate_image_dalle():
    try:
        summary_prompt = output_text_area.get("1.0", tk.END).strip()
        if not summary_prompt:
            messagebox.showerror("Error", "Please generate a summary first!")
            return

        if len(summary_prompt) > 1000:
            summary_prompt = summary_prompt[:1000]
            messagebox.showwarning("Warning", "Prompt truncated to 1000 characters.")

        set_buttons_state(tk.DISABLED)
        progress_bar.start()

        def run_image_gen():
            try:
                response = client.images.generate(
                    model="dall-e-2",
                    prompt=summary_prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = response.data[0].url
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                image = Image.open(BytesIO(img_response.content))
                image.save('generated_image_dalle.png')

                root.after(0, lambda: (
                    image.show(),
                    messagebox.showinfo("Success", "Image generated and saved as generated_image_dalle.png!"),
                    progress_bar.stop(),
                    set_buttons_state(tk.NORMAL)
                ))
            except Exception as e:
                root.after(0, lambda: (
                    messagebox.showerror("Error", f"Image generation failed: {str(e)}"),
                    progress_bar.stop(),
                    set_buttons_state(tk.NORMAL)
                ))

        threading.Thread(target=run_image_gen, daemon=True).start()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        progress_bar.stop()
        set_buttons_state(tk.NORMAL)


# Helper function to toggle button states
def set_buttons_state(state):
    generate_summary_button.config(state=state)
    text_to_speech_button.config(state=state)
    generate_image_dalle_button.config(state=state)


# GUI Layout
input_text_label = tk.Label(root, text="Input Text:")
input_text_label.pack(pady=5)
input_text_area = tk.Text(root, height=10, width=60)
input_text_area.pack(pady=5)

generate_summary_button = tk.Button(root, text="Generate Summary", command=generate_summary)
generate_summary_button.pack(pady=5)

output_text_label = tk.Label(root, text="Output Summary:")
output_text_label.pack(pady=5)
output_text_area = tk.Text(root, height=10, width=60)
output_text_area.pack(pady=5)

text_to_speech_button = tk.Button(root, text="Convert Summary to Speech", command=text_to_speech)
text_to_speech_button.pack(pady=5)

generate_image_dalle_button = tk.Button(root, text="Generate Image (DALL-E)", command=generate_image_dalle)
generate_image_dalle_button.pack(pady=5)

progress_bar = ttk.Progressbar(root, mode='indeterminate')
progress_bar.pack(pady=5)

# Start the main loop
root.mainloop()