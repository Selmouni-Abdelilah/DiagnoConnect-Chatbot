import sqlite3
from datetime import datetime
class DatabaseManager:

    def __init__(self, db_file):
        """
        Initialize the DatabaseManager.
        :param db_file: The name of the SQLite database file.
        """
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()
        self.create_database_table()

    def create_database_table(self):
        """
        Create the 'EmbeddingMetadata' table in the database if it doesn't exist.
        This table stores metadata about embedded blobs.
        :return: None
        """
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS EmbeddingMetadata
                              (BlobName TEXT PRIMARY KEY, ExistsInBlob TEXT, ExistsInSearch TEXT, LatestEmbeddedOn DATE)''')
        self.conn.commit()

    def record_exists_in_database(self, blob_name):
        """
        Check if a record with the given `blob_name` exists in the database.
        :param blob_name: The name of the blob to check.
        :return: True if the record exists, False otherwise.
        """
        self.cursor.execute("SELECT COUNT(*) FROM EmbeddingMetadata WHERE BlobName=?", (blob_name,))
        count = self.cursor.fetchone()[0]
        return count > 0

    def insert_record_to_database(self, blob_name, exists_in_blob, exists_in_search):
        """
        Insert a new record into the 'EmbeddingMetadata' table.
        :param blob_name: The name of the blob to insert.
        :param exists_in_blob: A flag indicating if the blob exists.
        :param exists_in_search: A flag indicating if the blob exists in search.
        :return: None
        """
        latest_embedded_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.record_exists_in_database(blob_name):
            self.cursor.execute("INSERT INTO EmbeddingMetadata (BlobName, ExistsInBlob, ExistsInSearch, LatestEmbeddedOn) VALUES (?, ?, ?, ?)",
                               (blob_name, exists_in_blob, exists_in_search, latest_embedded_on))
            self.conn.commit()