import csv
import re
import pprint

TEST_CASES = []

with open("data/test_csv.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)
    fourth_column = [row[4] for row in rows]
    procedures = []
    pattern = r"\d+º Código (\w+) Designação (.*?)(?=\d+º Código|$)"
    for i in fourth_column:
        

        matches = re.findall(pattern, i)

        

        for code, description in matches:
            procedures.append({
                "description": description.strip(),
                "expected_code": code,
                "notes": "Extraído automaticamente do CSV"
            })


with open("test_cases.py", "w", encoding="utf-8") as h:
    h.write("TEST_CASES = ")
    h.write(pprint.pformat(procedures, indent=4))
    h.write("\n")
print("finish")