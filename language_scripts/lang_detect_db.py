import sqlite3
from tqdm import tqdm
import langdetect


def detect_language(text):
    if text is None:
        return 'unknown'
    try:
        return langdetect.detect(text.strip()) if text.strip() else 'unknown'
    except langdetect.lang_detect_exception.LangDetectException:
        return 'unknown'


def check_and_add_column(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    if 'detected_language' not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN detected_language TEXT")

def update_languages(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the detected_language column exists and add if not
    check_and_add_column(cursor, table_name)

    # Fetch text data from the table where detected_language might be missing
    cursor.execute(f"SELECT id, text FROM {table_name} WHERE detected_language IS NULL OR detected_language = ''")
    text_data = cursor.fetchall()

    # Detect language and update the database
    for idx, (row_id, text) in tqdm(enumerate(text_data), total=len(text_data), desc=f"Updating {table_name}"):
        detected_language = detect_language(text)
        cursor.execute(f"UPDATE {table_name} SET detected_language = ? WHERE id = ?", (detected_language, row_id))

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print(f"Language detection and update completed for table {table_name}")


def main():
    db_path = '/Users/j_v_samson/Repos/inequality_classifier/inequality_data.sqlite'
    table_names = ['channel_results_20240416_1729', 'comment_results_20240416_1729', 'group_results_20240416_1729']

    for table in table_names:
        update_languages(db_path, table)


if __name__ == "__main__":
    main()
