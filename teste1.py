from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from typing import List, Tuple
import glob
import dlib

# Load a model
model = YOLO("bestMelhorIndiano.pt")
class_names = model.names

# Create frames directory if it doesn't exist
frames_dir = '/home/cauacampos/dev/MapSync-IA/frames'
os.makedirs(frames_dir, exist_ok=True)

def load_images(folder_path: str):
    """Load all images from a folder"""
    return glob.glob(os.path.join(folder_path, '*.[pj][pn]g'))

def load_videos(folder_path: str):
    """Load all videos from a folder"""
    return glob.glob(os.path.join(folder_path, '*.mp4'))

def process_image(img_path: str, show_result: bool = True):
    """Process a single image, detect potholes and count them"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return
    
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)

    pothole_count = 0
    detection_made = False

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            detection_made = True
            masks = masks.data.cpu().numpy()
            for seg, box in zip(masks, boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    pothole_count += 1
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)  # Blue color
                    cv2.putText(img, f"{c} {pothole_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if detection_made:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        frame_filename = os.path.join(frames_dir, f"detection_{timestamp}_{os.path.basename(img_path)}")
        cv2.imwrite(frame_filename, img)
        print(f"Saved detection frame: {frame_filename}")
        print(f"Total potholes detected: {pothole_count}")

    if show_result:
        cv2.imshow('Image Analysis', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pothole_count

def process_images(folder_path: str):
    """Process all images in a folder"""
    image_files = load_images(folder_path)
    total_potholes = 0
    
    for image_file in image_files:
        total_potholes += process_image(image_file)
    
    print(f"\nTotal potholes detected in all images: {total_potholes}")

def is_new_pothole(pothole: Tuple[int, int, int, int], detected_potholes: List[Tuple[int, int, int, int]]) -> bool:
    """Check if a pothole is new by comparing with previously detected potholes"""
    for dp in detected_potholes:
        if abs(pothole[0] - dp[0]) < 10 and abs(pothole[1] - dp[1]) < 10 and abs(pothole[2] - dp[2]) < 10 and abs(pothole[3] - dp[3]) < 10:
            return False
    return True

def merge_contours(contours: List[np.ndarray], min_dist: int = 10) -> List[np.ndarray]:
    """Merge contours that are close to each other"""
    merged_contours = []
    for contour in contours:
        merged = False
        for i, mc in enumerate(merged_contours):
            if cv2.pointPolygonTest(mc, (int(contour[0][0][0]), int(contour[0][0][1])), True) < min_dist:
                merged_contours[i] = np.vstack((mc, contour))
                merged = True
                break
        if not merged:
            merged_contours.append(contour)
    return merged_contours

def process_video(video_path: str):
    """Process a single video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    detection_count = 0
    total_potholes = 0
    playback_delay = 100  # Adjust this value to control playback speed (higher = slower)
    
    # Initialize dlib correlation trackers
    trackers = []
    detected_potholes = []  # To keep track of detected potholes
    next_id = 1

    while True:
        ret, img = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        img = cv2.resize(img, (1020, 500))
        h, w, _ = img.shape
        results = model.predict(img)

        detection_made = False
        current_frame_potholes = 0
        new_potholes = []

        for r in results:
            boxes = r.boxes
            masks = r.masks

            if masks is not None:
                detection_made = True
                masks = masks.data.cpu().numpy()
                for seg, box in zip(masks, boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = merge_contours(contours)  # Merge close contours

                    for contour in contours:
                        d = int(box.cls)
                        c = class_names[d]
                        x, y, w_rect, h_rect = cv2.boundingRect(contour)
                        pothole = (x, y, w_rect, h_rect)
                        
                        if is_new_pothole(pothole, detected_potholes):
                            new_potholes.append(pothole)
                            current_frame_potholes += 1
                            cv2.rectangle(img, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)  # Blue color
                            cv2.putText(img, f"{c}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if new_potholes:
            detected_potholes.extend(new_potholes)
            detection_count += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame_filename = os.path.join(frames_dir, f"detection_{timestamp}_{detection_count}.jpg")
            cv2.imwrite(frame_filename, img)
            print(f"Saved detection frame: {frame_filename}")
            print(f"New potholes in this frame: {current_frame_potholes}")

        cv2.imshow('Video Analysis - Press Q to quit', img)
        key = cv2.waitKey(playback_delay) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal potholes detected in video: {total_potholes}")

def process_videos(folder_path: str):
    """Process all videos in a folder"""
    video_files = load_videos(folder_path)
    total_potholes_all_videos = 0
    
    for video_file in video_files:
        print(f"\nProcessing video: {os.path.basename(video_file)}")
        total_potholes_all_videos += process_video(video_file)
    
    print(f"\nTotal potholes detected in all videos: {total_potholes_all_videos}")

def main():
    """Main function to handle user input and processing"""
    print("MapSync-IA Pothole Detection System")
    print("--------------------------------------")
    
    while True:
        choice = input("\nVocê quer analisar fotos ou vídeos? (fotos/videos/sair): ").strip().lower()
        
        if choice == 'sair':
            print("Encerrando o programa...")
            break
        elif choice == 'fotos':
            folder_path = '/home/cauacampos/dev/MapSync-IA/fotos'
            if not os.path.exists(folder_path):
                print(f"Pasta não encontrada: {folder_path}")
            else:
                process_images(folder_path)
        elif choice == 'videos':
            folder_path = '/home/cauacampos/dev/MapSync-IA/video'
            if not os.path.exists(folder_path):
                print(f"Pasta não encontrada: {folder_path}")
            else:
                process_videos(folder_path)
        else:
            print("Escolha inválida. Por favor, digite 'fotos', 'videos' ou 'sair'.")
            
if __name__ == "__main__":
    main()
