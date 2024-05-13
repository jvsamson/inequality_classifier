import pandas as pd
import torch
from sqlalchemy import create_engine, MetaData, Table, Column, Float, Integer, Text, BIGINT
from tqdm.auto import tqdm
from transformers import pipeline
from accelerate import Accelerator

# Initialize database connections
source_engine = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/detected_language.sqlite')
target_engine = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/processed_classified.sqlite')

# Reflect the source and target database schemas
source_metadata = MetaData()
source_metadata.reflect(bind=source_engine)
target_metadata = MetaData()
target_metadata.reflect(bind=target_engine)

# Initialize accelerator for MPS
accelerator = Accelerator(device_placement=True, cpu=True if not torch.backends.mps.is_available() else False)
print(f"Using device: {accelerator.device}")


# Classification function using Hugging Face transformers
def classify_text(text, multi_label):
    if not text.strip():
        return None  # No classification if text is empty

    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                          device=accelerator.device)
    class_prompts = {
        "top_bottom": "Is this text discussing socio-economic disparities such as wealth or power distribution, income inequality, or class struggles? Does it focus on material resources, standard of living, or the tension between different economic classes, within society?",
        "inside_outside": "Is this text discussing issues of national belonging, citizenship, immigration, or the distinctions between insiders and outsiders of a society? Is it about integration policies, debates on national identity, or the challenges of open versus closed borders?",
        "us_them": "Is this text centered on identity-based discrimination or struggles for recognition, particularly involving characteristics like gender, race, or ethnicity? Does it address issues of social recognition and the challenges faced by specific groups based on nationality, gender, or racial identity?",
        "today_tomorrow": "Is this text focused on ecological sustainability, climate change, and the impact of current decisions on future generations? Does it discuss the temporal aspects of inequality in relation to socio-economic decision making, environmental policies, climate actions, and their long-term effects?"
    }
    results = classifier(text, list(class_prompts.values()), multi_label=multi_label)
    return {f"{'multi' if multi_label else 'single'}_{key}": value for key, value in
            zip(class_prompts.keys(), results['scores'])}


# Process data from a table, classify texts, and save the results in a new table
def process_and_classify(source_engine, target_engine, source_table, text_column, new_table_name):
    data = pd.read_sql(f"SELECT * FROM {source_table} WHERE detected_language = 'en';", source_engine)
    total_batches = len(data) // 500 + (len(data) % 500 != 0)

    if new_table_name not in target_metadata.tables:
        columns_from_original = [Column(c.name, BIGINT if isinstance(c.type, BIGINT) else Text) for c in
                                 source_metadata.tables[source_table].columns]
        classification_columns = [Column(f"{label}_{key}", Float) for label in ['multi', 'single'] for key in
                                  ['top_bottom', 'inside_outside', 'us_them', 'today_tomorrow']]
        new_table = Table(new_table_name, target_metadata, *columns_from_original, *classification_columns)
        new_table.create(bind=target_engine)

    pbar_batches = tqdm(total=total_batches, desc=f"Classifying texts in {source_table}")
    for start in range(0, len(data), 500):
        end = start + 500
        batch = data.iloc[start:end].copy()
        batch['multi_results'] = batch[text_column].apply(lambda x: classify_text(x, True))
        batch['single_results'] = batch[text_column].apply(lambda x: classify_text(x, False))

        for classification_type in ['multi_results', 'single_results']:
            for key in ['top_bottom', 'inside_outside', 'us_them', 'today_tomorrow']:
                batch[f"{classification_type.split('_')[0]}_{key}"] = batch[classification_type].apply(
                    lambda x: x[f"{classification_type.split('_')[0]}_{key}"] if x else None)

        columns_to_insert = [c.name for c in target_metadata.tables[new_table_name].columns]
        batch = batch[columns_to_insert]
        batch.to_sql(new_table_name, con=target_engine, if_exists='append', index=False)
        pbar_batches.update(1)

    pbar_batches.close()
    print(f"Data from {source_table} classified and saved in {new_table_name}")


# Main function to execute the process
def main():
    process_and_classify(source_engine, target_engine, 'group_processed', 'text', 'group_processed_twice')


if __name__ == "__main__":
    main()
