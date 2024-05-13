import os
import langdetect
from openai import OpenAI
from tqdm import tqdm

# Set API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def translate_to_english_openai(data):
    translations = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Translating"):
        text = row['text']
        detected_lang = row.get('detected_language', None)

        if not text.strip():
            translations.append({'index': index, 'translation': '', 'language': 'unknown'})
            continue

        # Use pre-detected language if available
        if detected_lang:
            if detected_lang == 'en':
                translations.append({'index': index, 'translation': text, 'language': detected_lang})
                continue
        else:
            # Perform language detection if not already done
            try:
                detected_lang = langdetect.detect(text)
                if detected_lang == 'en':
                    translations.append({'index': index, 'translation': text, 'language': detected_lang})
                    continue
            except langdetect.lang_detect_exception.LangDetectException:
                detected_lang = 'unknown'

        # Translate the text if necessary
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You will be provided with a Text in an unkonwn language, and your task is to "
                               "translate it into English."
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            temperature=0.5,
            max_tokens=64,
            top_p=1
        )
        translation = response.choices[0].message.content.strip()
        translations.append({'index': index, 'translation': translation, 'language': detected_lang})

    return translations
