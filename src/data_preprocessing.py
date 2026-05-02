import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MultiLabelBinarizer

# Paths
REPORTS_DIR = os.path.join("data", "reports", "ecgen-radiology")
IMAGES_DIR = os.path.join("data", "images")
PROCESSED_DIR = os.path.join("data", "processed")

def parse_xml_report(xml_path):
    """Parses a single Open-I XML report."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract images associated with this report
    image_ids = []
    for parentImage in root.findall(".//parentImage"):
        image_id = parentImage.attrib.get('id')
        if image_id:
            image_ids.append(image_id + ".png")
            
    # Extract findings and impression
    findings = ""
    impression = ""
    for AbstractText in root.findall(".//AbstractText"):
        label = AbstractText.attrib.get('Label')
        text = AbstractText.text if AbstractText.text else ""
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text
            
    # Combine text for the final report text
    report_text = f"FINDINGS: {findings} IMPRESSION: {impression}".strip()
    
    # Extract MeSH labels (Diagnoses)
    labels = []
    for mesh in root.findall(".//MeSH"):
        major = mesh.find("major")
        if major is not None and major.text:
            labels.append(major.text.lower())
    
    # If no labels found, mark as "normal"
    if not labels:
        labels.append("normal")
        
    return image_ids, report_text, labels

def create_dataset():
    """Iterates through reports, matches with images, and creates a dataset."""
    print("Parsing reports and building dataset...")
    xml_files = glob.glob(os.path.join(REPORTS_DIR, "*.xml"))
    
    if not xml_files:
        print("No XML reports found in data/reports/")
        return
        
    dataset_records = []
    
    for xml_path in xml_files:
        image_ids, report_text, labels = parse_xml_report(xml_path)
        
        # Open-I often has multiple images per report. We'll map each image to the report.
        for image_name in image_ids:
            image_path = os.path.join(IMAGES_DIR, image_name)
            
            # Check if image actually exists in our folder
            if os.path.exists(image_path):
                dataset_records.append({
                    "image_path": image_path,
                    "report_text": report_text,
                    "labels": labels
                })
                
    if not dataset_records:
        print("No matching images found for the reports. Please check your data folders.")
        return
        
    df = pd.DataFrame(dataset_records)
    
    # --- UPGRADE: Hierarchical Class Filtering (Top 15 Classes) ---
    print("Filtering classes to Top 15 most frequent diseases to solve zero-prediction collapse...")
    from collections import Counter
    all_labels = [label for labels in df['labels'] for label in labels]
    label_counts = Counter(all_labels)
    
    top_15_classes = [label for label, count in label_counts.most_common(15)]
    print(f"Top 15 Classes: {top_15_classes}")
    
    def filter_top_classes(labels):
        filtered = [l for l in labels if l in top_15_classes]
        # If nothing left, default to normal if it's in top 15, else drop it
        return filtered if filtered else ["normal"] if "normal" in top_15_classes else []
        
    df['labels'] = df['labels'].apply(filter_top_classes)
    df = df[df['labels'].map(len) > 0] # Remove any records that became empty
    # --------------------------------------------------------------
    
    # Multi-label Binarization for the labels
    print("Binarizing labels...")
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(df['labels'])
    
    # Save the label classes for future reference
    classes = mlb.classes_
    pd.DataFrame({"class_name": classes}).to_csv(os.path.join(PROCESSED_DIR, "labels.csv"), index=False)
    print(f"Found {len(classes)} unique classes.")
    
    # Add binarized columns to the dataframe
    for i, class_name in enumerate(classes):
        df[class_name] = labels_encoded[:, i]
        
    # We drop the raw list format and save to CSV
    df = df.drop(columns=['labels'])
    dataset_out_path = os.path.join(PROCESSED_DIR, "dataset.csv")
    df.to_csv(dataset_out_path, index=False)
    
    print(f"Dataset successfully created at {dataset_out_path} with {len(df)} records!")

if __name__ == "__main__":
    create_dataset()
