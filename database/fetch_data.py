from database.credentials import DatabaseCredentials
import pandas as pd
import mysql.connector


def fetch_table_data(table_name, row_limit=None):
    creds = DatabaseCredentials()
    connection = creds.create_database_connection()

    limit_clause = f"LIMIT {row_limit}" if row_limit is not None else ""
    query = f"SELECT *, '{table_name}' as source_table FROM {table_name} ORDER BY RAND() {limit_clause};"

    try:
        connection.connect()
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        return pd.DataFrame(rows)
    except mysql.connector.Error as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            connection.close()


def get_combined_data(row_limit=10):
    channel_data = fetch_table_data('channel_results', row_limit)
    comment_data = fetch_table_data('comment_results', row_limit)
    group_data = fetch_table_data('group_results', row_limit)

    combined_data = pd.concat([channel_data, comment_data, group_data], ignore_index=True)
    return combined_data, channel_data, comment_data, group_data
