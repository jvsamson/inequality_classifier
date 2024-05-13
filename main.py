import os
import pandas as pd
from tqdm import tqdm
from database.fetch_data import get_combined_data, fetch_table_data
from language_scripts.translator_m2m100 import translate_to_english_m2m100
from language_scripts.translator_mbart import translate_to_english_mbart
from language_scripts.translator_openai import translate_to_english_openai
from classification_scripts.inequality_classifier import classify_text


def process_data(data, translation_func, classification_columns, filter_lang=None):
    processed_data = []
    try:
        for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing rows"):
            if filter_lang and row.get('detected_language') != filter_lang:
                continue
            text = str(row['text'])
            print(f"Processing row {index}, Post ID: {row['id']} from Table: {row['source_table']}")
            if text.strip():
                translation_result = translation_func(data.loc[data.index == index])
                translated_text = translation_result[0]['translation']
                detected_lang = translation_result[0]['language']

                if translated_text.strip():
                    classification_scores = classify_text(translated_text, classification_columns)
                    for column in classification_columns:
                        row[column] = classification_scores.get(column, 0)
                processed_data.append(row)
    except KeyboardInterrupt:
        print("Interrupted during data processing. Saving what has been processed so far.")
    return pd.DataFrame(processed_data)


def main(use_csv=False, csv_path=None, process_combined=False, row_limit=None, translation_choice="openai",
         filter_lang=None):
    translation_funcs = {
        "openai": translate_to_english_openai,
        "mbart": translate_to_english_mbart,
        "m2m100": translate_to_english_m2m100
    }
    translation_func = translation_funcs.get(translation_choice)
    classification_columns = ['top_bottom_inequalities', 'inside_outside_inequalities', 'us_them_inequalities',
                              'today_tomorrow_inequalities']

    try:
        if use_csv:
            print("Processing data from CSV...")
            data = pd.read_csv(csv_path)
            if row_limit is not None:
                data = data.sample(n=row_limit, random_state=1)
            base_name = os.path.basename(csv_path).split('.')[0]
        else:
            if process_combined:
                print("Fetching combined data from database...")
                data, channel_data, comment_data, group_data = get_combined_data(row_limit)
                base_name = "combined_data"
            else:
                table_names = ['channel_results', 'comment_results', 'group_results']
                processed_data = pd.DataFrame()
                for table_name in table_names:
                    print(f"Fetching {table_name} from database...")
                    data = fetch_table_data(table_name, row_limit)
                    temp_processed = process_data(data, translation_func, classification_columns, filter_lang)
                    processed_data = pd.concat([processed_data, temp_processed])
                base_name = "individual_tables"

        processed_data = process_data(data, translation_func, classification_columns, filter_lang)

    except KeyboardInterrupt:
        print("Interrupted! Saving processed data so far...")
    finally:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        filename = f"{base_name}_processed_{timestamp}.csv"
        processed_data.to_csv(filename, index=False)
        print(f"Processed data saved to {filename}")


if __name__ == "__main__":
    main(use_csv=True,
         csv_path="data/channel_lang_20240412_2029.csv",
         process_combined=True,
         row_limit=None,
         translation_choice="openai",
         filter_lang='en')
