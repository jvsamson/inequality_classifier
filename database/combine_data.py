import pandas as pd
from sqlalchemy import create_engine

# Initialize database connections
engine_processed_classified = create_engine('sqlite:////Users/j_v_samson/Repos/inequality_classifier/processed_classified.sqlite')
engine_comment_finished = create_engine('sqlite:////Users/j_v_samson/Desktop/comment_finished.sqlite')


# Function to process and merge tables
def merge_tables():
    # Load data from each table
    df_group = pd.read_sql_table('group_processed_twice', con=engine_processed_classified)
    df_channel = pd.read_sql_table('channel_classified_twice', con=engine_processed_classified)
    df_comment = pd.read_sql_table('comment_classified', con=engine_comment_finished)

    # Add 'typ' column
    df_group['typ'] = 'group'
    df_channel['typ'] = 'channel'
    df_comment['typ'] = 'comment'

    # Define the columns we need, ensuring we handle missing columns appropriately
    desired_columns = ['id', 'typ', 'detected_language', 'text', 'english_text',
                       'multi_top_bottom', 'multi_inside_outside', 'multi_us_them', 'multi_today_tomorrow',
                       'single_top_bottom', 'single_inside_outside', 'single_us_them', 'single_today_tomorrow']

    # Create a function to select columns safely
    def safe_select(df, columns):
        # Select existing columns and fill missing ones with None
        existing_columns = df.columns.intersection(columns)
        missing_columns = [col for col in columns if col not in existing_columns]
        df = df[existing_columns].copy()
        for col in missing_columns:
            df[col] = None
        return df[columns]  # Return columns in the specified order

    # Apply the safe select function
    df_group = safe_select(df_group, desired_columns)
    df_channel = safe_select(df_channel, desired_columns)
    df_comment = safe_select(df_comment, desired_columns)

    # Merge dataframes
    df_final = pd.concat([df_group, df_channel, df_comment], ignore_index=True)

    # Save the merged dataframe to a new table in the 'processed_classified' database
    df_final.to_sql('classified_data', con=engine_processed_classified, if_exists='replace', index=False)

    print("Tables merged and saved successfully.")


# Execute the function
if __name__ == "__main__":
    merge_tables()
