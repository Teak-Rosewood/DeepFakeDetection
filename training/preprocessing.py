import json
import os

# Path to the directory containing the metadata.json and the video files
data_dir = './data/dfdc_train_part_0'
metadata_path = os.path.join(data_dir, 'metadata.json')

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Example structure of metadata
for key, value in metadata.items():
    print(f"Filename: {key}, Label: {value['label']}, Original: {value['original']}")

detector = MTCNN()

def crop_faces_single_frame(image_path, output_dir):
    base_filename = os.path.basename(image_path)
    print(base_filename)
    cropped_filename = f"cropped_0_{base_filename}"
    cropped_filepath = os.path.join(output_dir, cropped_filename)
    
    image = cv2.imread(image_path)
    faces = detector.detect_faces(image)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        cropped_face = image[y:y+h, x:x+w]
        face_filename = f"cropped_{i}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, face_filename), cropped_face)

def crop_faces_parallel(frame_dir, cropped_dir):
    frame_files = []
    for subdir, dirs, files in os.walk(frame_dir):
        for file in files:
            if file.endswith('.jpg'):
                frame_files.append(os.path.join(subdir, file))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(crop_faces_single_frame, frame_file, cropped_dir) for frame_file in frame_files]
        for future in futures:
            future.result()
crop_faces_parallel(frame_dir=frames_dir, cropped_dir=faces_dir)