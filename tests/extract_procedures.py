import csv
import re
import pprint

TEST_CASES = []

with open("data/test_csv.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)
    third_column = [row[3] for row in rows]
    fourth_column = [row[4] for row in rows]
    # print(fourth_column[1])
    procedures = []
    pattern = r"\d+º Código (\w+) Designação (.*?)(?=\d+º Código|$)"
    # matches = re.findall(pattern, fourth_column[1], re.DOTALL)
    # print(matches)
    # for code, description in matches:
    #     print(code)
    #     print(description.strip())
    #     procedures.append({
    #             "description": description.strip(),
    #             "expected_code": code,
    #             "notes": "Extraído automaticamente do CSV"
    #         })
    # print(procedures)
    for i in fourth_column:
        

        matches = re.findall(pattern, i, re.DOTALL)

        

        for code, description in matches:
            procedures.append({
                "description": description.strip(),
                "expected_code": code,
                "notes": "Extraído automaticamente do CSV"
            })

    for i in third_column:
        

        matches = re.findall(pattern, i, re.DOTALL)

        

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