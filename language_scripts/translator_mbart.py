from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langdetect import detect
import re
import torch
from tqdm import tqdm

model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def clean_text(text):
    # Function to remove emojis and special characters
    return re.sub(r'[^\w\s,.\-?!]', '', text)


def map_language_code(lang):
    # Mapping based on provided list, add more as needed
    mbart_lang_codes = {
        "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX",
        "et": "et_EE", "fi": "fi_FI", "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN",
        "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "ko": "ko_KR", "lt": "lt_LT",
        "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO",
        "ru": "ru_RU", "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", "zh": "zh_CN",
        "af": "af_ZA", "az": "az_AZ", "bn": "bn_IN", "fa": "fa_IR", "he": "he_IL",
        "hr": "hr_HR", "id": "id_ID", "ka": "ka_GE", "km": "km_KH", "mk": "mk_MK",
        "ml": "ml_IN", "mn": "mn_MN", "mr": "mr_IN", "pl": "pl_PL", "ps": "ps_AF",
        "pt": "pt_XX", "sv": "sv_SE", "sw": "sw_KE", "ta": "ta_IN", "te": "te_IN",
        "th": "th_TH", "tl": "tl_XX", "uk": "uk_UA", "ur": "ur_PK", "xh": "xh_ZA",
        "gl": "gl_ES", "sl": "sl_SI"
    }
    return mbart_lang_codes.get(lang, None)


def translate_to_english_mbart(texts):
    translations = []
    for text in tqdm(texts, desc="Translating"):
        try:
            text = clean_text(text)
            if not text.strip():
                translations.append(('', 'unknown'))
                continue

            detected_lang = detect(text)
            src_lang_code = map_language_code(detected_lang)

            if src_lang_code is None:
                print(f"Unsupported language for translation: {detected_lang}")
                translations.append((text, detected_lang))
                continue

            tokenizer.src_lang = src_lang_code
            max_model_length = model.config.max_position_embeddings
            encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                                max_length=max_model_length).to(device)
            generated_tokens = model.generate(**encoded)
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translations.append((translation, detected_lang))
        except Exception as e:
            print(f"Error during translation: {e}, Text: {text}, Source Language: {detected_lang}")
            translations.append(('', detected_lang))

    return translations
