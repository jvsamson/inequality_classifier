import sys

from database.fetch_data import get_combined_data
import langdetect
from tqdm import tqdm


def detect_language(texts):
    language_detection_results = []
    for text in tqdm(texts, desc="Detecting Languages"):
        try:
            if text is None or not text.strip():
                language_detection_results.append('unknown')
                continue
            detected_lang = langdetect.detect(text)
            language_detection_results.append(detected_lang)
        except langdetect.lang_detect_exception.LangDetectException:
            language_detection_results.append('unknown')
    return language_detection_results


def process_and_save_data_with_language(data, data_type):
    print(f"Detecting languages for {data_type}...")
    data['detected_language'] = detect_language(data['text'].tolist())

    # Creating a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f"{data_type}_lang_{timestamp}.csv"
    data.to_csv(file_name, index=False)
    print(f"Data saved to {file_name}")


def main():
    # row_limit = None if len(sys.argv) > 1 and sys.argv[1] == '--all' else 10
    row_limit = None
    print("Fetching data...")
    _, channel_data, comment_data, group_data = get_combined_data(row_limit=row_limit)

    print("Processing individual tables with language detection...")
    process_and_save_data_with_language(channel_data, 'channel')
    process_and_save_data_with_language(comment_data, 'comment')
    process_and_save_data_with_language(group_data, 'group')

if __name__ == "__main__":
    main()
