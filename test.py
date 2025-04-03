from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

# Load a model
model = YOLO("bestMelhorIndiano.pt")
class_names = model.names

# Create frames directory if it doesn't exist
frames_dir = '/home/cauacampos/dev/MapSync-AI/frames'
os.makedirs(frames_dir, exist_ok=True)

def process_images():
    folder_path = '/home/cauacampos/dev/MapSync-AI/fotos'
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1020, 500))
        h, w, _ = img.shape
        results = model.predict(img)

        for r in results:
            boxes = r.boxes
            masks = r.masks

            if masks is not None:
                masks = masks.data.cpu()
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        d = int(box.cls)
                        c = class_names[d]
                        x, y, x1, y1 = cv2.boundingRect(contour)
                        cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                        cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def process_videos():
    folder_path = '/home/cauacampos/dev/MapSync-AI/video'
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detection_count = 0
        
        # Set slower playback speed (milliseconds between frames)
        playback_delay = 100  # Adjust this value to control playback speed (higher = slower)

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
            
            for r in results:
                boxes = r.boxes
                masks = r.masks

                if masks is not None:
                    detection_made = True
                    masks = masks.data.cpu()
                    for seg, box in zip(masks.data.cpu().numpy(), boxes):
                        seg = cv2.resize(seg, (w, h))
                        contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for contour in contours:
                            d = int(box.cls)
                            c = class_names[d]
                            x, y, x1, y1 = cv2.boundingRect(contour)
                            cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                            cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save frame if detection was made
            if detection_made:
                detection_count += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                frame_filename = os.path.join(frames_dir, f"detection_{timestamp}_{detection_count}.jpg")
                cv2.imwrite(frame_filename, img)
                print(f"Saved detection frame: {frame_filename}")

            cv2.imshow('Video Analysis - Press Q to quit', img)
            
            # Add delay for slower playback
            key = cv2.waitKey(playback_delay) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    choice = input("Você quer analisar fotos ou vídeos? (fotos/videos): ").strip().lower()
    if choice == 'fotos':
        process_images()
    elif choice == 'videos':
        process_videos()
    else:
        print("Escolha inválida. Por favor, digite 'fotos' ou 'videos'.")

if __name__ == "__main__":
    main()