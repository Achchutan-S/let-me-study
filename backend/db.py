import os
import certifi
import motor.motor_asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise RuntimeError('MONGODB_URI not configured')

MONGO_INSECURE = os.getenv('MONGO_INSECURE', 'false').lower() == 'true'

if MONGO_INSECURE:
    client = motor.motor_asyncio.AsyncIOMotorClient(
        MONGODB_URI,
        tlsAllowInvalidCertificates=True,
    )
else:
    client = motor.motor_asyncio.AsyncIOMotorClient(
        MONGODB_URI,
        tlsCAFile=certifi.where(),
    )

# Determine DB name: prefer path segment; else env MONGODB_DB; else default
_db_name = os.getenv('MONGODB_DB')
if not _db_name:
    parsed = urlparse(MONGODB_URI)
    path = (parsed.path or '').lstrip('/')
    _db_name = path if path else 'exam_helper'

db = client.get_database(_db_name)


async def get_database():
    return db
