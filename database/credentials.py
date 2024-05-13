import os
from dotenv import load_dotenv
import mysql.connector
import mysql.connector.errors
from mysql.connector import MySQLConnection


class DatabaseCredentials:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Access environment variables
        self.hostname: str = os.environ["SQL_HOSTNAME"]
        self.port: str = os.environ["SQL_PORT"]
        self.database: str = os.environ["SQL_DATABASE"]
        self.username: str = os.environ["SQL_USERNAME"]
        self.password: str = os.environ["SQL_PASSWORD"]

    def create_database_connection(self) -> MySQLConnection:
        try:
            db_connection = mysql.connector.connect(
                host=self.hostname,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )

            if db_connection.is_connected():
                print("Database connection works!")
                db_connection.close()

            return db_connection

        except mysql.connector.Error as error:
            print(f'{error}')
