import json
from typing import Dict
import requests

EXCHANGE_URL = "https://api.exchangerate-api.com/v4/latest/RUB"

def fetch_exchange_rates() -> Dict[str, float]:
    """Parse exchange rates for RUB, USD, EUR"""

    try:
        response = requests.get(EXCHANGE_URL)
        rates = response.json()["rates"]
        rates["RUR"] = rates["RUB"]
        return rates
    except requests.exceptions.SSLError as e:
        raise AssertionError("Cannot get exchange rate! Try later or change the host API") from e

if __name__ == "__main__":
    rates = fetch_exchange_rates()
    print(rates)
