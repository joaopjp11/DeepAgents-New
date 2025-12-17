import csv

with open("data/test_csv.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)
    fourth_column = [row[4] for row in rows]
    print(fourth_column[1])
