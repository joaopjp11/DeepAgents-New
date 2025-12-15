import xml.etree.ElementTree as ET
import csv
from collections import defaultdict


INPUT_XML = r"C:\Users\joaop\ICD10\ICD10\icd10pcs_tables_2026.xml"
OUTPUT_CSV = r"data/icd10pcs_tables_2026.csv"

SECTION_CODES = {
    "Medical and Surgical": "0",
    "Obstetrics": "1",
    "Placement": "2",
    "Administration": "3",
    "Measurement and Monitoring": "4",
    "Extracorporeal or Systemic Assistance and Performance": "5",
    "Extracorporeal or Systemic Therapies": "6",
    "Osteopathic": "7",
    "Other Procedures": "8",
    "Chiropractic": "9",
    "Imaging": "B",
    "Nuclear Medicine": "C",
    "Radiation Therapy": "D",
    "Physical Rehabilitation and Diagnostic Audiology": "F",
    "Mental Health": "G",
    "Substance Abuse Treatment": "H",
    "New Technology": "X",
}


def parse_tables_xml():
    """
    Parse ICD-10-PCS tabular structure.
    """
    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    rows = []
    components_data = defaultdict(lambda: defaultdict(set))

    for table_idx, pcs_table in enumerate(root.findall("pcsTable")):
        section = ""
        section_code = ""
        body_system = ""
        body_system_code = ""
        operation = ""
        operation_code = ""
        operation_def = ""

        for axis in pcs_table.findall("axis[@pos]"):
            pos = axis.get("pos")
            title = axis.find("title")
            title_text = title.text if title is not None else ""

            labels = axis.findall("label")
            if labels:
                code = labels[0].get("code", "")
                label_text = labels[0].text if labels[0].text else ""
                definition = axis.find("definition")
                definition_text = definition.text if definition is not None else ""

                if pos == "1":
                    section = label_text
                    section_code = SECTION_CODES.get(section, "0")
                elif pos == "2":
                    body_system = label_text
                    body_system_code = code
                elif pos == "3":
                    operation = label_text
                    operation_code = code
                    operation_def = definition_text

                components_data[pos][title_text].add((code, label_text))

        for row in pcs_table.findall("pcsRow"):
            codes_attr = row.get("codes", "0") 

            body_part_options = []
            approach_options = []
            device_options = []
            qualifier_options = []

            for axis in row.findall("axis[@pos]"):
                pos = axis.get("pos")
                title = axis.find("title")
                title_text = title.text if title is not None else ""

                labels = axis.findall("label")
                label_list = [(l.get("code", ""), l.text if l.text else "") for l in labels]

                for code, label_text in label_list:
                    components_data[pos][title_text].add((code, label_text))

                if pos == "4":
                    body_part_options = label_list
                elif pos == "5":
                    approach_options = label_list
                elif pos == "6":
                    device_options = label_list
                elif pos == "7":
                    qualifier_options = label_list

            for bp_code, bp_label in body_part_options:
                for ap_code, ap_label in approach_options:
                    for dv_code, dv_label in device_options:
                        for ql_code, ql_label in qualifier_options:
                            full_code = f"{section_code}{body_system_code}{operation_code}{bp_code}{ap_code}{dv_code}{ql_code}"
                            rows.append({
                                "full_code": full_code,
                                "section": section,
                                "body_system": body_system,
                                "operation": operation,
                                "operation_definition": operation_def,
                                "body_part_code": bp_code,
                                "body_part": bp_label,
                                "approach_code": ap_code,
                                "approach": ap_label,
                                "device_code": dv_code,
                                "device": dv_label,
                                "qualifier_code": ql_code,
                                "qualifier": ql_label,
                            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "full_code",
                "section",
                "body_system",
                "operation",
                "operation_definition",
                "body_part_code",
                "body_part",
                "approach_code",
                "approach",
                "device_code",
                "device",
                "qualifier_code",
                "qualifier",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parse_tables_xml()