import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Global setup for the directory path
base_dir = Path('/analysis_tables')
base_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# Database setup
engine = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/processed_classified.sqlite')


# Fetch data from the database, format it, and organize into tables
def fetch_and_organize_data(thresholds, categories, types):
    results = {}

    # Fetch total counts for normalization
    total_counts = {}
    for typ in ['overall'] + types:
        query_total = f"SELECT COUNT(*) as total FROM classified_data WHERE typ = '{typ}'" if typ != 'overall' else "SELECT COUNT(*) as total FROM classified_data;"
        total_counts[typ] = pd.read_sql_query(query_total, con=engine).iloc[0, 0]

    # Create tables for each type and overall
    for typ in ['overall'] + types:
        for threshold in thresholds:
            row_data = {category: None for category in categories}
            for category in categories:
                where_clause = f"WHERE typ = '{typ}' AND" if typ != "overall" else "WHERE"
                query = f"SELECT COUNT(*) FROM classified_data {where_clause} {category} >= {threshold};"
                count = pd.read_sql_query(query, con=engine).iloc[0, 0]
                proportion = (count / total_counts[typ] * 100) if total_counts[typ] else 0
                formatted_output = f"{count:,} ({proportion:,.2f}%)" if count > 0 else "-"
                row_data[category] = formatted_output
            if typ not in results:
                results[typ] = []
            results[typ].append(row_data)

    # Convert the list of dictionaries to DataFrames
    for typ, data in results.items():
        df = pd.DataFrame(data, index=thresholds)
        results[typ] = df

        # Save individual CSV files
        df.to_csv(base_dir / f'{typ}_data.csv')

    return results


# Main function to handle the workflow
def main():
    categories = [
        'multi_top_bottom', 'multi_inside_outside', 'multi_us_them', 'multi_today_tomorrow',
        'single_top_bottom', 'single_inside_outside', 'single_us_them', 'single_today_tomorrow'
    ]
    thresholds = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
    types = ['group', 'channel', 'comment']

    # Generate and save the data
    data_tables = fetch_and_organize_data(thresholds, categories, types)

    # Export to a single Excel file with different tabs
    with pd.ExcelWriter(base_dir / 'analysis.xlsx') as writer:
        for key, df in data_tables.items():
            df.to_excel(writer, sheet_name=key)


if __name__ == "__main__":
    main()
