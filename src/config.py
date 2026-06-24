from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
USER_DOCS = DATA_DIR / 'userdocs'
VECTOR_DIR = BASE_DIR / 'vector_store'
MODEL = 'gpt-4o-mini'
