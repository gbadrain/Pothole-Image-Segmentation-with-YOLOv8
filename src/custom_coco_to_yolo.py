import json
import os
from collections import defaultdict

def convert_coco_to_yolo_segmentation(coco_json_path, output_dir):
    """
    Converts COCO segmentation annotations to YOLO segmentation format.

    Args:
        coco_json_path (str): Path to the COCO annotation JSON file.
        output_dir (str): Directory where YOLO labels and class names will be saved.
    """
    print(f"Starting conversion from COCO: {coco_json_path} to YOLO format in: {output_dir}")

    # Create output directories
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Prepare mappings
    image_id_to_info = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']}
                        for img in coco_data['images']}

    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create a mapping from original COCO category_id to new 0-indexed YOLO class_id
    # This ensures class IDs are contiguous and 0-indexed for YOLO
    unique_category_ids = sorted(list(category_id_to_name.keys()))
    coco_id_to_yolo_id = {original_id: new_id for new_id, original_id in enumerate(unique_category_ids)}

    # Save class names to a file
    class_names_path = os.path.join(output_dir, 'classes.txt')
    with open(class_names_path, 'w') as f:
        for original_id in unique_category_ids:
            f.write(f"{category_id_to_name[original_id]}\n")
    print(f"Class names saved to: {class_names_path}")

    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Process each image's annotations
    for image_id, image_info in image_id_to_info.items():
        file_name = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']

        yolo_lines = []
        for ann in annotations_by_image[image_id]:
            category_id = ann['category_id']
            segmentations = ann['segmentation'] # This is a list of polygons

            # Convert COCO category_id to YOLO class_id
            yolo_class_id = coco_id_to_yolo_id.get(category_id)
            if yolo_class_id is None:
                print(f"Warning: Category ID {category_id} not found in unique categories. Skipping annotation for image {file_name}.")
                continue

            # COCO segmentation can be a list of polygons for a single instance
            # Each polygon needs to be normalized and added to the YOLO line
            for segment in segmentations:
                # Flatten the list of coordinates and normalize
                normalized_segment = []
                for i in range(0, len(segment), 2):
                    x = segment[i]
                    y = segment[i+1]
                    normalized_x = x / img_width
                    normalized_y = y / img_height
                    normalized_segment.append(f"{normalized_x:.6f}")
                    normalized_segment.append(f"{normalized_y:.6f}")

                # Construct the YOLO line
                yolo_line = f"{yolo_class_id} " + " ".join(normalized_segment)
                yolo_lines.append(yolo_line)

        # Write YOLO annotations to file
        if yolo_lines:
            output_label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + '.txt')
            with open(output_label_path, 'w') as f:
                for line in yolo_lines:
                    f.write(line + '\n')
        else:
            print(f"No annotations found for image: {file_name}. Skipping label file creation.")

    print("Conversion complete!")

if __name__ == "__main__":
    # Example Usage (replace with your actual paths)
    train_coco_json_file = '/Users/GURU/pothole-segmentation/data/train/_annotations.coco.json'
    train_yolo_output_folder = '/Users/GURU/pothole-segmentation/data/train'
    convert_coco_to_yolo_segmentation(train_coco_json_file, train_yolo_output_folder)

    valid_coco_json_file = '/Users/GURU/pothole-segmentation/data/valid/_annotations.coco.json'
    valid_yolo_output_folder = '/Users/GURU/pothole-segmentation/data/valid'
    convert_coco_to_yolo_segmentation(valid_coco_json_file, valid_yolo_output_folder)
