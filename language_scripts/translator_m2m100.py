from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect
import torch
import re
from tqdm import tqdm

model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# Check if MPS (Apple Silicon GPU) is available and set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    # Check if a CUDA-enabled GPU is available and set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)


def clean_text(text):
    # Function to remove emojis and special characters
    return re.sub(r'[^\w\s,.\-?!]', '', text)


def translate_to_english_m2m100(texts):
    translations = []
    for text in tqdm(texts, desc="Translating"):
        try:
            text = clean_text(text)
            # Ensure text is not empty
            if not text.strip():
                translations.append(('', 'unknown'))
                continue

            # Check if the text is already in English
            if detect(text) == 'en':
                translations.append((text, 'en'))
                continue

            # Tokenize and truncate/pad the text
            detected_lang = detect(text)
            tokenizer.src_lang = detected_lang
            encoded_inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)  # Adjust max_length as needed
            encoded_inputs = encoded_inputs.to(device)

            # Generate the translation
            generated_tokens = model.generate(**encoded_inputs)
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translations.append(translation)
        except Exception as e:
            print(f"Skipping text due to error: {e}, Text: {text}, Source Language: {detected_lang}")
            translations.append(('', detected_lang))  # Append empty string or handle the error as needed
    return translations
