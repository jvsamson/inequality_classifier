import sqlite3
import pandas as pd
import mysql.connector
from datetime import datetime
from database.credentials import DatabaseCredentials


def fetch_table_data(table_name, sqlite_db_path, row_limit=None):
    creds = DatabaseCredentials()
    connection = creds.create_database_connection()
    limit_clause = f"LIMIT {row_limit}" if row_limit else ""
    query = f"SELECT * FROM {table_name} {limit_clause};"

    try:
        connection.connect()
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        data = pd.DataFrame(rows)

        # Saving directly to SQLite database
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        table_name_with_timestamp = f"{table_name}_{timestamp}"
        with sqlite3.connect(sqlite_db_path) as conn:
            data.to_sql(name=table_name_with_timestamp, con=conn, if_exists='replace', index=False)
            print(f"Data from {table_name} inserted into SQLite table {table_name_with_timestamp}")
    except mysql.connector.Error as e:
        print(f"Error fetching data from MySQL: {e}")
    finally:
        if connection.is_connected():
            connection.close()


def main():
    sqlite_db_path = '/Users/j_v_samson/Repos/inequality_classifier/inequality_data.sqlite'
    tables = ['channel_results', 'comment_results', 'group_results']
    for table in tables:
        fetch_table_data(table, sqlite_db_path, row_limit=None)  # Adjust row_limit as needed


if __name__ == "__main__":
    main()
