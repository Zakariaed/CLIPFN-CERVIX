import pandas as pd
import os

# Path to your cervix dataset
dataset_path = 'cervix_data_224'

# Class descriptions for cervical histopathology
class_descriptions = {
    "HSIL": "High-grade Squamous Intraepithelial Lesion: severe dysplasia with marked nuclear abnormalities, increased mitotic activity, and loss of cellular maturation in upper epithelial layers.",
    "LSIL": "Low-grade Squamous Intraepithelial Lesion: mild dysplasia with enlarged nuclei, irregular chromatin, and koilocytotic changes primarily in superficial epithelial layers.",
    "NIM": "Normal/Negative for Intraepithelial Malignancy: normal cervical epithelium with regular cellular architecture, uniform nuclear morphology, and appropriate maturation.",
    "SCC": "Squamous Cell Carcinoma: invasive malignant epithelial cells with pleomorphic nuclei, prominent nucleoli, and invasion through basement membrane into underlying stroma."
}

all_files = []
texts = []
labels = []

# Enumerate through class folders
for label_idx, class_name in enumerate(class_descriptions.keys()):
    class_path = os.path.join(dataset_path, class_name)
    
    # Check if class folder exists
    if not os.path.exists(class_path):
        print(f"Warning: Folder {class_path} does not exist. Skipping...")
        continue
    
    # Get all image files (supporting multiple formats)
    files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
             if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    
    all_files.extend(files)
    texts.extend([class_descriptions[class_name]] * len(files))
    labels.extend([label_idx] * len(files))
    
    print(f"Found {len(files)} images in {class_name} class")

# Create DataFrame
df = pd.DataFrame({
    'images': all_files,
    'text': texts,
    'labels': labels
})

# Save to CSV
output_csv = './cervix_data_224.csv'
df.to_csv(output_csv, index=False)

print(f"\nDataset processing complete!")
print(f"Total images processed: {len(df)}")
print(f"Classes: {list(class_descriptions.keys())}")
print(f"CSV saved to {output_csv}")

# Display class distribution
class_counts = df.groupby('labels').size()
print(f"\nClass distribution:")
for idx, class_name in enumerate(class_descriptions.keys()):
    count = class_counts.get(idx, 0)
    print(f"  {class_name}: {count} images")
