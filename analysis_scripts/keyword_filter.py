import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Float, Text, BIGINT
from tqdm.auto import tqdm

# Initialize database connection
engine = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/processed_classified.sqlite')

# Reflect the database schema
metadata = MetaData()
metadata.reflect(bind=engine)


# Function to filter rows based on keywords and add a column indicating the keyword
def filter_and_add_keyword_column(data, keywords):
    new_rows = []  # List to store new rows with their keywords

    for _, row in data.iterrows():
        row['text'] = row['text'] if row['text'] is not None else ''
        row['english_text'] = row['english_text'] if row['english_text'] is not None else ''
        keywords_found = set()  # Using a set to avoid duplicating keywords

        for keyword, word_list in keywords.items():
            if any(word in row['text'] or word in row['english_text'] for word in word_list):
                keywords_found.add(keyword)

        for keyword in keywords_found:
            new_row = row.copy()  # Copy the original row
            new_row['keyword'] = keyword  # Add the keyword column
            new_rows.append(new_row)

    # Convert the list of new rows into a DataFrame
    if new_rows:
        return pd.DataFrame(new_rows)
    else:
        return pd.DataFrame(columns=data.columns.append(pd.Index(['keyword'])))  # Ensure the keyword column is included even in empty DF


# Process data from a table, filter based on keywords, and save the results in a new table
def process_and_filter(source_table, new_table_name, keywords):
    data = pd.read_sql(f"SELECT * FROM {source_table};", engine)
    total_batches = len(data) // 500 + (len(data) % 500 != 0)

    if new_table_name not in metadata.tables:
        columns_from_original = [Column(c.name, BIGINT if isinstance(c.type, BIGINT) else Text) for c in
                                 metadata.tables[source_table].columns]
        new_table = Table(new_table_name, metadata, *columns_from_original, Column('keyword', Text))
        new_table.create(bind=engine)

    pbar_batches = tqdm(total=total_batches, desc=f"Filtering texts in {source_table}")
    for start in range(0, len(data), 500):
        end = start + 500
        batch = data.iloc[start:end].copy()
        filtered_batch = filter_and_add_keyword_column(batch, keywords)
        filtered_batch.to_sql(new_table_name, con=engine, if_exists='append', index=False)
        pbar_batches.update(1)

    pbar_batches.close()
    print(f"Data from {source_table} filtered and saved in {new_table_name}")


# Main function to execute the process
def main():
    keywords = {
        'Top-Bottom': ['elite', 'power', 'wealth', 'rent', 'lease', 'work', 'price', 'inflation', 'pension',
                       'retire', 'annuity', 'poor', 'oppressed', 'income', 'salary', 'class', 'disparity',
                       'distribution', 'standard of living', 'econom', 'privil', 'inequality', 'affluent',
                       'marginalized', 'status', 'stasi', 'poverty', 'rich', 'capitalism', 'socialism',
                       'bourgeois', 'proletariat', 'homeless', 'welfare', 'corrupt', 'exploit', 'redistrib'],
        'Inside-Outside': ['foreign', 'national', 'immigrant', 'citizen', 'outsider', 'migrant', 'border',
                           'integration', 'refug', 'asylum', 'cultur', 'identity', 'replace', 'diaspora',
                           'muslim', 'jew', 'islam', 'exile', 'alien', 'headscarf', 'minority', 'reset', 'resettlement', ],
        'Us-Them': ['gender', 'race', 'racism', 'racial', 'queer', 'feminism', 'ethni', 'discrimination', 'identity', 'minor', 'equal',
                    'prejudice', 'stereotype', 'diversity', 'inclusion', 'sexism', 'homo', 'xeno', 'LGB',
                    'indigenous', 'bias', 'political correctness', 'language police', 'semitism', 'solidar', 'intersec'],
        'Today-Tomorrow': ['future', 'fff', 'fridaysforfuture', 'thunberg', 'last generation', 'clima',
                           'sustain', 'generation', 'environ', 'ecolog', 'carbon', 'footprint', 'warming',
                           'greenhouse', 'renewable', 'energy', 'conservation', 'pollut', 'deforest',
                           'biodivers', 'emissions']
    }

    process_and_filter('classified_data', 'filtered_data', keywords)


if __name__ == "__main__":
    main()
