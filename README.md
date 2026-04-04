# EMS data analysis

- API endpoint: https://data.cityofnewyork.us/resource/76xm-jjuj.json
- OpenAPI documentation: https://dev.socrata.com/foundry/data.cityofnewyork.us/76xm-jjuj
- Codebook: [ems_codebook.json](ems_codebook.json)

# Get the Data into a local SQLite database

``` bash
# Full load (expect a long wait)
python3 fetch_ems_to_sqlite.py
```

Database is located at ems_incidents.sqlite.