import requests, yaml


def load_openapi_spec(url: str) -> dict:
    r = requests.get(url)
    r.raise_for_status()
    return yaml.safe_load(r.text)

spec = load_openapi_spec("https://documentation.ubuntu.com/lxd/latest/rest-api.yaml")

