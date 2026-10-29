
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("DATABASE_URL")
if not url:
    raise SystemExit("DATABASE_URL not found. Is your .env file in this folder, named exactly .env?")

conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute("SELECT version();")
print("Connected! Database says:", cur.fetchone()[0])
cur.close()
conn.close()