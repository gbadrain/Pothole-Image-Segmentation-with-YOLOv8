from ultralytics.data.converter import convert_coco
import traceback

# Define the path to your COCO annotations directory
# This should be the directory containing your _annotations.coco.json files
# For example, if your structure is data/train/_annotations.coco.json
# and data/valid/_annotations.coco.json, you'd point to data/train and data/valid

# Assuming your data structure is data/train and data/valid
# The converter will look for _annotations.coco.json within these directories

try:
    print("Converting train annotations...")
    convert_coco(labels_dir='/Users/GURU/pothole-segmentation/data/train', use_segments=True)
except Exception as e:
    print(f"Error converting train annotations: {e}")
    traceback.print_exc()

try:
    print("Converting valid annotations...")
    convert_coco(labels_dir='/Users/GURU/pothole-segmentation/data/valid', use_segments=True)
except Exception as e:
    print(f"Error converting valid annotations: {e}")
    traceback.print_exc()

print("Conversion attempt complete.")