import pandas as pd
import torch
from sqlalchemy import create_engine
from tqdm.auto import tqdm
from transformers import pipeline
from accelerate import Accelerator

# Initialize database connections
engine_save = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/processed_classified.sqlite')

# Initialize accelerator for MPS
accelerator = Accelerator(device_placement=True, cpu=True if not torch.backends.mps.is_available() else False)
print(f"Using device: {accelerator.device}")


# Classification function using Hugging Face transformers for single-label classification
def classify_text(text, multi_label):
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
    results = classifier(text, list(class_prompts.values()), multi_label=multi_label)
    return {prompt: score for prompt, score in zip(class_prompts.keys(), results['scores'])}


# Process data from a table, classify texts, and save the results in a new table
def process_and_classify(engine, table_name, text_column, new_table_name):
    data = pd.read_sql(f"SELECT *, text as text_copy FROM {table_name} WHERE detected_language = 'en';", engine)
    total_batches = len(data) // 500 + (len(data) % 500 != 0)

    pbar_batches = tqdm(total=total_batches, desc=f"Classifying texts in {table_name}")
    for start in range(0, len(data), 500):
        end = start + 500
        batch = data.iloc[start:end]
        batch_classification_sl = batch['text_copy'].apply(lambda x: classify_text(x, multi_label=False))
        for column in batch_classification_sl.iloc[0].keys():
            batch.loc[start:end - 1, column + '_sl'] = batch_classification_sl.apply(
                lambda x: x[column] if x is not None else None)

        # Drop temporary copy used for classification
        batch.drop(columns=['text_copy'], inplace=True)

        batch.to_sql(new_table_name, con=engine_save, if_exists='append', index=False)
        pbar_batches.update(1)  # Increment progress bar
    pbar_batches.close()
    print(f"Data from {table_name} classified and saved in {new_table_name}")


# Main function to re-process and classify the texts
def main():
    # Re-process the already classified table with single-label classification
    process_and_classify(engine_save, 'channel_classified', 'text', 'channel_classified_twice')


if __name__ == "__main__":
    main()
