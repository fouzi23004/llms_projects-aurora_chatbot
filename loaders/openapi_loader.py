import requests, yaml
import os
from dotenv import load_dotenv


def load_openapi_spec(url: str) -> dict:
    r = requests.get(url)
    r.raise_for_status()
    return yaml.safe_load(r.text)

load_dotenv()
UBUNTU_DOC_URL = os.getenv("UBUNTU_DOC_URL")
OPENAPI_SPEC_FILE = os.getenv("OPENAPI_SPEC_FILE")

OPENAPI_SPEC_URL = f"{UBUNTU_DOC_URL}{OPENAPI_SPEC_FILE}"

spec = load_openapi_spec(OPENAPI_SPEC_URL)