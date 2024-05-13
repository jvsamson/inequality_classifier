import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Set up the directory path for saving the analysis results.
base_dir = Path('/Users/j_v_samson/Repos/inequality_classifier/analysis_tables')
base_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# Set up the database connection.
engine = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/processed_classified.sqlite')


# Function to analyze data by keyword group and classification category.
def analyze_by_keyword_group(keyword_group, thresholds, categories, source_table, total_dataset_size):
    results = {}
    # Fetch all entries for the given keyword group from the specified table.
    query = f"SELECT * FROM {source_table} WHERE keyword = '{keyword_group}';"
    data = pd.read_sql(query, engine)

    # Ensure all classification columns are numeric; non-numeric entries are converted to NaN.
    for category in categories:
        data[category] = pd.to_numeric(data[category], errors='coerce')

    # Count the total entries that have the specific keyword.
    total_entries_keyword = data.shape[0]

    # Analyze data by category and threshold.
    for category in categories:
        results[category] = []
        for threshold in thresholds:
            # Count entries that exceed the current threshold.
            count = data[data[category] >= threshold].dropna().shape[0]
            # Calculate the percentage of the total dataset that this count represents.
            proportion_total = (count / total_dataset_size * 100) if total_dataset_size > 0 else 0
            # Calculate the percentage of the keyword group that this count represents.
            proportion_keyword = (count / total_entries_keyword * 100) if total_entries_keyword > 0 else 0
            formatted_output = f"{count:,} ({proportion_total:.2f}%, {proportion_keyword:.2f}%)" if count > 0 else "-"
            results[category].append(formatted_output)

    # Convert the results dictionary into a DataFrame for easier manipulation and display.
    df = pd.DataFrame(results, index=thresholds)
    return df


def main():
    # Define the categories and thresholds for classification comparison.
    categories = ['multi_top_bottom', 'multi_inside_outside', 'multi_us_them', 'multi_today_tomorrow',
                  'single_top_bottom', 'single_inside_outside', 'single_us_them', 'single_today_tomorrow']
    thresholds = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
    filtered_table = 'filtered_data'
    source_table = 'classified_data'
    keyword_groups = ['Top-Bottom', 'Inside-Outside', 'Us-Them', 'Today-Tomorrow']

    # Calculate the total number of entries in the complete dataset for normalization.
    total_dataset_size = pd.read_sql(f"SELECT COUNT(*) as total FROM {source_table};", engine).iloc[0]['total']

    # Prepare an Excel writer to save each analysis in a different tab.
    excel_path = base_dir / 'keyword_analysis.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        # Perform analysis for each keyword group and save it in the Excel file.
        for group in keyword_groups:
            df = analyze_by_keyword_group(group, thresholds, categories, filtered_table, total_dataset_size)
            df.to_excel(writer, sheet_name=group)
            print(f"Analysis for {group} saved in the sheet: {group}")

    print(f"All analyses have been saved successfully to {excel_path}")


if __name__ == "__main__":
    main()
