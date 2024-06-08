
import os
import zipfile
import json
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from tkinter import Tk, filedialog, ttk, messagebox, StringVar, Radiobutton, Button, Label, Frame
from shutil import copy2
from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define a list of colors for different classes
COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "cyan", "magenta", "lime",
    "maroon", "navy", "olive", "teal", "violet", "brown", "indigo", "coral", "gold", "silver"
]

# Function to extract the zip file
def extract_zip_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to create segmentation data from semantic segmentation image
def create_segmentation(image_path, label_pixel_values):
    image = Image.open(image_path)
    image_array = np.array(image)
    segmentation = {}

    for label, pixel_value in label_pixel_values.items():
        mask = np.all(image_array == pixel_value, axis=-1).astype(np.uint8)
        if np.sum(mask) > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            segmentation[label] = []
            for contour in contours:
                if len(contour) > 2:  # Ensure the contour has at least 3 points
                    segmentation[label].append(contour.flatten().tolist())

    return segmentation

# Function to convert SOLO data to COCO format
def convert_solo_to_coco(zip_file_path, output_path, coco_format_type, progress_callback):
    extract_to = 'extracted_solo_data'
    extract_zip_file(zip_file_path, extract_to)
    
    solo_folder = os.path.join(extract_to, 'solo_23')
    sequence_folder = os.path.join(solo_folder, 'sequence.0')
    
    annotation_definitions_path = os.path.join(solo_folder, 'annotation_definitions.json')
    with open(annotation_definitions_path, 'r') as file:
        annotation_definitions = json.load(file)
    
    coco_format = {
        "info": {
            "description": f"Converted SOLO to COCO ({coco_format_type})",
            "version": "1.0",
            "year": 2024,
            "contributor": "User",
            "date_created": "2024-06-08"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Default License",
                "url": "http://example.com"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = []
    for item in annotation_definitions['annotationDefinitions'][0]['spec']:
        categories.append({
            "id": item['label_id'],
            "name": item['label_name'],
            "supercategory": "none"
        })

    coco_format['categories'] = categories

    image_id = 0
    annotation_id = 0

    frame_files = [file for file in os.listdir(sequence_folder) if file.endswith('.json')]
    total_files = len(frame_files)

    for index, frame_file in enumerate(frame_files):
        frame_data_path = os.path.join(sequence_folder, frame_file)
        
        with open(frame_data_path, 'r') as file:
            frame_data = json.load(file)
        
        capture = frame_data['captures'][0]
        image_info = {
            "id": image_id,
            "file_name": capture['filename'],
            "width": int(capture['dimension'][0]),
            "height": int(capture['dimension'][1]),
            "date_captured": "2024-06-08",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        
        coco_format['images'].append(image_info)
        
        bbox_annotations = capture['annotations'][0]['values']
        for bbox in bbox_annotations:
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": bbox['labelId'],
                "bbox": [
                    bbox['origin'][0],
                    bbox['origin'][1],
                    bbox['dimension'][0],
                    bbox['dimension'][1]
                ],
                "area": bbox['dimension'][0] * bbox['dimension'][1],
                "segmentation": [],
                "iscrowd": 0
            }
            coco_format['annotations'].append(annotation_info)
            annotation_id += 1
        
        segmentation_image_path = os.path.join(sequence_folder, capture['annotations'][1]['filename'])
        segmentation_instances = capture['annotations'][1]['instances']
        label_pixel_values = {inst['labelName']: inst['pixelValue'] for inst in segmentation_instances}
        
        segmentation_data = create_segmentation(segmentation_image_path, label_pixel_values)
        
        for label, polygons in segmentation_data.items():
            for ann in coco_format['annotations']:
                if categories[ann['category_id'] - 1]['name'] == label:
                    ann['segmentation'] = polygons
        
        image_id += 1

        # Update progress
        progress_callback((index + 1) / total_files * 100)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Copy images to the output path
    for image_info in coco_format['images']:
        source_image_path = os.path.join(sequence_folder, image_info['file_name'])
        destination_image_path = os.path.join(output_path, image_info['file_name'])
        copy2(source_image_path, destination_image_path)
    
    # Minify and save the COCO formatted JSON
    output_json_path = os.path.join(output_path, f'solo_to_coco_with_segmentation_{coco_format_type}.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file, separators=(',', ':'))
    
    return output_json_path

# Function to update progress
def update_progress(value, progress_bar):
    progress_bar['value'] = value

# Function to handle the long-running task
def long_running_task(zip_file_path, output_path, coco_format_type, progress_bar, progress_window):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(convert_solo_to_coco, zip_file_path, output_path, coco_format_type, lambda v: update_progress(v, progress_bar))
        result = future.result()
        progress_window.after(0, lambda: complete_and_close(progress_window, result))

# Function to show completion message and close the window
def complete_and_close(progress_window, result):
    messagebox.showinfo("Completed", f"Conversion completed. Output saved to: {result}")
    progress_window.destroy()

# Function to load and visualize annotation for a single image
def load_and_visualize():
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    annotation_path = filedialog.askopenfilename(title="Select JSON Annotation File", filetypes=[("JSON files", "*.json")])
    
    if not image_path or not annotation_path:
        return
    
    with open(annotation_path, 'r') as file:
        annotations = json.load(file)
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Find the relevant annotations for the image
    image_id = next((img['id'] for img in annotations['images'] if img['file_name'] == os.path.basename(image_path)), None)
    if image_id is None:
        messagebox.showerror("Error", "No annotations found for the selected image.")
        return
    
    image_annotations = [anno for anno in annotations['annotations'] if anno['image_id'] == image_id]
    
    # Create a mini window to display class names with colors
    class_window = Tk()
    class_window.title("Class Colors")
    class_frame = Frame(class_window)
    class_frame.pack(pady=10)
    
    category_colors = {cat['id']: COLORS[i % len(COLORS)] for i, cat in enumerate(annotations['categories'])}
    
    for cat in annotations['categories']:
        color = category_colors[cat['id']]
        Label(class_frame, text=cat['name'], bg=color, width=20).pack(anchor='w', padx=5, pady=2)

    class_window.update()
    
    for anno in image_annotations:
        color = category_colors[anno['category_id']]
        for seg in anno['segmentation']:
            poly = np.array(seg).reshape((len(seg) // 2, 2))
            draw.polygon([tuple(p) for p in poly], outline=color)
    
    image.show()

# Function to open file dialog and start the conversion process
def start_conversion():
    def start_conversion_task():
        zip_file_path = filedialog.askopenfilename(title="Select ZIP file", filetypes=[("ZIP files", "*.zip")])
        if not zip_file_path:
            return

        output_path = filedialog.askdirectory(title="Select Output Directory")
        if not output_path:
            return

        progress_window = Tk()
        progress_window.title("Converting SOLO to COCO")
        progress_bar = ttk
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(pady=20)

        threading.Thread(target=long_running_task, args=(zip_file_path, output_path, coco_format_type.get(), progress_bar, progress_window)).start()
        progress_window.mainloop()

    root = Tk()
    root.title("Choose COCO Format")
    
    coco_format_type = StringVar(value="segmentation")
    
    Radiobutton(root, text="Segmentation", variable=coco_format_type, value="segmentation").pack(anchor='w')
    Radiobutton(root, text="Bounding Box", variable=coco_format_type, value="bbox").pack(anchor='w')
    Radiobutton(root, text="Keypoint", variable=coco_format_type, value="keypoint").pack(anchor='w')
    
    Button(root, text="Start Conversion", command=start_conversion_task).pack(pady=10)
    Button(root, text="Visualize Annotations", command=load_and_visualize).pack(pady=10)
    
    root.mainloop()

# Run the conversion process
if __name__ == "__main__":
    start_conversion()
