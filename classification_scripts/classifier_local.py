import pandas as pd
import torch
from sqlalchemy import create_engine
from tqdm.auto import tqdm
from transformers import pipeline
from accelerate import Accelerator

# Initialize database connections
engine_detected = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/detected_language.sqlite')
engine_comment = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/comment_translated.sqlite')
engine_save = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/processed_classified.sqlite')

# Initialize accelerator for MPS
accelerator = Accelerator(device_placement=True, cpu=True if not torch.backends.mps.is_available() else False)
print(f"Using device: {accelerator.device}")


# Classification function using Hugging Face transformers
def classify_text(text):
    if not text.strip():
        return None  # No classification if text is empty

    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                          device=accelerator.device)
    class_prompts = {
        "top_bottom_inequalities": "Is this text discussing socio-economic disparities such as wealth or power distribution, income inequality, or class struggles? Does it focus on material resources, standard of living, or the tension between different economic classes, within society?",
        "inside_outside_inequalities": "Is this text discussing issues of national belonging, citizenship, immigration, or the distinctions between insiders and outsiders of a society? Is it about integration policies, debates on national identity, or the challenges of open versus closed borders?",
        "us_them_inequalities": "Is this text centered on identity-based discrimination or struggles for recognition, particularly involving characteristics like gender, race, or ethnicity? Does it address issues of social recognition and the challenges faced by specific groups based on nationality, gender, or racial identity?",
        "today_tomorrow_inequalities": "Is this text focused on ecological sustainability, climate change, and the impact of current decisions on future generations? Does it discuss the temporal aspects of inequality in relation to socio-economic decision making, environmental policies, climate actions, and their long-term effects?"
    }
    results = classifier(text, list(class_prompts.values()), multi_label=True)
    return {prompt: score for prompt, score in zip(class_prompts.keys(), results['scores'])}


# Process data from a table, classify texts, and save the results in a new table
def process_and_classify(engine, table_name, text_column, new_table_name):
    data = pd.read_sql(f"SELECT * FROM {table_name} WHERE detected_language = 'en';", engine)
    total_batches = len(data) // 500 + (len(data) % 500 != 0)

    pbar_tables = tqdm(total=3, desc="Tables Processed")
    for table_index, table_name in enumerate([engine_detected, engine_comment, engine_save]):  # Assumed 3 tables
        pbar_tables.update(1)  # Increment outer progress bar

        pbar_batches = tqdm(total=total_batches, desc=f"Classifying texts in {table_name}", leave=False, position=0)
        for start in range(0, len(data), 500):
            end = start + 500
            batch = data.iloc[start:end]
            batch_progress_desc = f"Batch {start // 500 + 1}/{total_batches}"
            pbar_batch = tqdm(total=len(batch), desc=batch_progress_desc, leave=False, position=1)
            classification_results = batch[text_column].apply(classify_text)
            for column in classification_results.iloc[0].keys():  # Assuming the first row returns all keys
                batch[column] = classification_results.apply(lambda x: x[column] if x is not None else None)
            batch.to_sql(new_table_name, con=engine_save, if_exists='append', index=False)
            pbar_batch.update(len(batch))  # Increment inner progress bar to completion
            pbar_batch.close()
            pbar_batches.update(1)  # Increment outer progress bar
        pbar_batches.close()

    pbar_tables.close()
    print(f"Data from {table_name} classified and saved in {new_table_name}")


# Main function to process all specified tables
def main():
    # Tables from detected_language.sqlite
    process_and_classify(engine_detected, 'channel_processed', 'text', 'channel_classified')
    process_and_classify(engine_detected, 'group_processed', 'text', 'group_classified')

    # Table from comment_translated.sqlite
    process_and_classify(engine_comment, 'comment_results_20240416_1729_processed', 'english_text',
                         'comment_classified')


if __name__ == "__main__":
    main()
