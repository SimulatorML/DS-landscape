"""
Simple module to get current convertation rates RUB to other currencies

Exsample of using:
    fetch_exchange_rates()

    returns:
        {'RUB': 1,
        'AED': 0.0456,
        'AFN': 1.09,
        'ALL': 1.29 ...
    
"""



from typing import Dict
import requests

EXCHANGE_URL = "https://api.exchangerate-api.com/v4/latest/RUB"

def fetch_exchange_rates() -> Dict[str, float]:
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
