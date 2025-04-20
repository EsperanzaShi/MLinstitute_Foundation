import os
import psycopg2
from psycopg2 import OperationalError, sql


def get_db_connection():
    """
    Establish and return a new PostgreSQL database connection using env vars.
    Raises RuntimeError on failure.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            dbname=os.getenv('DB_NAME', 'mnist'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'example')
        )
        return conn
    except OperationalError as e:
        raise RuntimeError(f"Database connection failed: {e}")


def log_prediction(prediction: int, true_label: int):
    """
    Insert a new prediction record into the `predictions` table.
    """
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO predictions (ts, predicted, true_label) VALUES (NOW(), %s, %s)"
                    ),
                    (prediction, true_label)
                )
    except Exception as e:
        # You could log this exception to a file or monitoring system
        raise RuntimeError(f"Failed to log prediction: {e}")
    finally:
        conn.close()