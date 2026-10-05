# Price Data

## Bundled sample

A small subset of 2024 German ENTSO-E day-ahead prices is included at
`sample_de_2024_prices.csv` for running tests and examples without a
network connection.

## Using your own price data

Pass any hourly price series as a CSV or Excel file.  The loader picks
the first numeric column and trims or pads to exactly 8 760 hours.

```python
with open("my_prices.csv", "rb") as f:
    result = optimize_bess_arbitrage(price_bytes=f.read())
```

## Downloading full ENTSO-E data

1. Go to <https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show>
2. Select country, year, and resolution = **PT60M**
3. Export as CSV
4. Pass the downloaded file via `price_bytes`

## File format requirements

| Column | Type | Unit |
|---|---|---|
| Any name | numeric | EUR/MWh |

Only the **first numeric column** is used.  All other columns are ignored.