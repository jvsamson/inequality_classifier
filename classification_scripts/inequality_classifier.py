import torch
from transformers import pipeline


def classify_text(text, classification_columns):
    if not text.strip():  # Check if text is empty or whitespace
        print("Received empty text for classification.")
        return {column: 0 for column in classification_columns}

    top_bottom_inequalities = "Is this text discussing socio-economic disparities such as wealth or power distribution, income inequality, or class struggles? Does it focus on material resources, standard of living, or the tension between different economic classes, within society?"
    inside_outside_inequalities = "Is this text dealing with issues of national belonging, citizenship, immigration, or the distinctions between insiders and outsiders of a society? Is it about integration policies, debates on national identity, or the challenges of open versus closed borders?"
    us_them_inequalities = "Is this text centered on identity-based discrimination or struggles for recognition, particularly involving characteristics like gender, race, or ethnicity? Does it address issues of social recognition and the challenges faced by specific groups based on nationality, gender, or racial identity?"
    today_tomorrow_inequalities = "Is this text focused on ecological sustainability, climate change, and the impact of current decisions on future generations? Does it discuss the temporal aspects of inequality in relation to socio-economic decision making, environmental policies, climate actions, and their long-term effects?"
    classes_verbalized = [top_bottom_inequalities,
                          inside_outside_inequalities,
                          us_them_inequalities,
                          today_tomorrow_inequalities]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    zeroshot_classifier = pipeline("zero-shot-classification",
                                   model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                                   device=device)

    print("Classifying text...")
    output = zeroshot_classifier(text, classes_verbalized,  multi_label=False)
    scores = {classification_columns[i]: output['scores'][i] for i in range(len(classes_verbalized))}

    return scores


# Sample usage
# 'classification_columns = ['top_bottom_inequalities', 'inside_outside_inequalities', 'us_them_inequalities',
#                          'today_tomorrow_inequalities']
# sample_text = "Your sample text goes here"
# classification_result = classify_text(sample_text, classification_columns)
# print(classification_result)'


def classify_text(text, classification_columns, multi_label=False):
    if not text.strip():  # Check if text is empty
        return {column: 0 for column in classification_columns}

    # Classification prompts
    class_prompts = {
        "top_bottom_inequalities": "Is this text discussing socio-economic disparities such as wealth or power distribution, income inequality, or class struggles? Does it focus on material resources, standard of living, or the tension between different economic classes, within society?",
        "inside_outside_inequalities": "Is this text dealing with issues of national belonging, citizenship, immigration, or the distinctions between insiders and outsiders of a society? Is it about integration policies, debates on national identity, or the challenges of open versus closed borders?",
        "us_them_inequalities": "Is this text centered on identity-based discrimination or struggles for recognition, particularly involving characteristics like gender, race, or ethnicity? Does it address issues of social recognition and the challenges faced by specific groups based on nationality, gender, or racial identity?",
        "today_tomorrow_inequalities": "Is this text focused on ecological sustainability, climate change, and the impact of current decisions on future generations? Does it discuss the temporal aspects of inequality in relation to socio-economic decision making, environmental policies, climate actions, and their long-term effects?"
    }

    # Use the GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                          device=device)

    # Classify text
    results = classifier(text, list(class_prompts.values()), multi_label=multi_label)
    return {classification_columns[i]: results['scores'][i] for i in range(len(class_prompts))}


def process_table_and_classify(engine, table_name, multi_label=False):
    # Fetch English texts only
    query = f"SELECT * FROM {table_name} WHERE detected_language = 'en';"
    data = pd.read_sql(query, engine)

    # Prepare for classification
    classification_columns = list(class_prompts.keys())
    tqdm.pandas(desc="Classifying texts")
    data['classification_results'] = data['text'].progress_apply(
        lambda x: classify_text(x, classification_columns, multi_label))

    # Save the processed data
    new_table_name = f"{table_name}_classified_{'multi' if multi_label else 'single'}"
    data.to_sql(new_table_name, con=engine, if_exists='replace', index=False)
    print(f"Data classified and saved in {new_table_name}")


# Example usage
if __name__ == "__main__":
    table_names = ['comment_results_20240416_1729', 'channel_results_20240416_1729', 'group_results_20240416_1737']
    for table in table_names:
        process_table_and_classify(engine, table, multi_label=True)
