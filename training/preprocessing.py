import json
import os
import cv2
from multiprocessing import Pool, cpu_count
# # Path to the directory containing the metadata.json and the video files
# data_dir = './data/dfdc_train_part_0'
# metadata_path = os.path.join(data_dir, 'metadata.json')

# # Load metadata
# with open(metadata_path, 'r') as f:
#     metadata = json.load(f)

# # Example structure of metadata
# for key, value in metadata.items():
#     print(f"Filename: {key}, Label: {value['label']}, Original: {value['original']}")

# detector = MTCNN()

# def crop_faces_single_frame(image_path, output_dir):
#     base_filename = os.path.basename(image_path)
#     print(base_filename)
#     cropped_filename = f"cropped_0_{base_filename}"
#     cropped_filepath = os.path.join(output_dir, cropped_filename)
    
#     image = cv2.imread(image_path)
#     faces = detector.detect_faces(image)
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for i, face in enumerate(faces):
#         x, y, w, h = face['box']
#         cropped_face = image[y:y+h, x:x+w]
#         face_filename = f"cropped_{i}_{base_filename}"
#         cv2.imwrite(os.path.join(output_dir, face_filename), cropped_face)

# def crop_faces_parallel(frame_dir, cropped_dir):
#     frame_files = []
#     for subdir, dirs, files in os.walk(frame_dir):
#         for file in files:
#             if file.endswith('.jpg'):
#                 frame_files.append(os.path.join(subdir, file))

#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(crop_faces_single_frame, frame_file, cropped_dir) for frame_file in frame_files]
#         for future in futures:
#             future.result()
# crop_faces_parallel(frame_dir=frames_dir, cropped_dir=faces_dir)

# Preprocessing - Extract frames from videos

def extract_frames_single_video(args):
    video_path, output_dir, fps = args
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_dir = os.path.join(output_dir, video_name)
    
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % fps == 0:
            frame_id = int(count / fps)
            cv2.imwrite(os.path.join(output_video_dir, f"frame_{frame_id}.jpg"), image)
        success, image = vidcap.read()
        count += 1

def extract_frames_parallel(video_dir, output_dir, fps=1):
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    args = [(video, output_dir, fps) for video in video_files]

    with Pool(cpu_count()) as pool:
        pool.map(extract_frames_single_video, args)

# extract_frames_parallel(video_dir=data_dir, output_dir=frames_dir, fps=2)