#%%

import csv

growth_rates = {}

with open("worldbank_wdi_pop.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Only process rows where Series Name indicates annual population growth
        if row["Series Name"] == "Population growth (annual %)":
            country_code = row["Country Code"]
            
            # Collect all numeric values from 2010 to 2023
            values = []
            for year in range(2010, 2024):
                col_name = f"{year} [YR{year}]"
                raw_val = row[col_name].strip()
                if raw_val not in ["", ".."]:
                    try:
                        values.append(float(raw_val))
                    except ValueError:
                        pass
            
            # Compute the average of available years
            if values:
                avg_growth = sum(values) / len(values)
            else:
                avg_growth = 0.0
            
            growth_rates[country_code] = avg_growth

# Sort by country code alphabetically
sorted_codes = sorted(growth_rates.keys())

# Write output to a text file in the requested format
with open("growth_rates.txt", "w", encoding="utf-8") as out:
    out.write("{\n")
    for code in sorted_codes:
        out.write(f"\"{code}\": {round(growth_rates[code], 4)},\n")
    out.write("}\n")

#%%