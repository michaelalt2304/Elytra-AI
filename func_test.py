import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO

# Load model
model = YOLO("models/best_v11.pt")

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Define MiDaS preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),  # Resize for MiDaS model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def process_video_for_depth(video_source=0):
    # Open video file or webcam
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

        # Run YOLO detection
        results = model(frame_rgb)

        # Run MiDaS depth estimation
        img_input = transform(frame_rgb).unsqueeze(0)
        with torch.no_grad():
            depth_map = midas(img_input).squeeze().cpu().numpy()

        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # Get depth map dimensions
        depth_h, depth_w = depth_map.shape
        orig_h, orig_w, _ = frame.shape

        object_id = 0  

        for r in results:
            boxes = r.boxes.xyxy  
            class_ids = r.boxes.cls  

            for i, class_id in enumerate(class_ids):
                if int(class_id) == 0:  
                    object_id += 1
                    x_min, y_min, x_max, y_max = map(int, boxes[i])  # Bounding box coordinates
                    
                    # Compute object center
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                    # Scale coordinates for depth map
                    x_scaled = int((x_center / orig_w) * depth_w)
                    y_scaled = int((y_center / orig_h) * depth_h)

                    # Ensure valid coordinates
                    x_scaled = np.clip(x_scaled, 0, depth_w - 1)
                    y_scaled = np.clip(y_scaled, 0, depth_h - 1)

                    # Get relative depth from MiDaS
                    estimated_depth_relative = depth_map[y_scaled, x_scaled]

                    # Draw bounding box on frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    cv2.putText(frame, f"Trash {object_id} ({estimated_depth_relative:.2f})", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 255, 0), 2, cv2.LINE_AA)

        
        cv2.imshow("Detected Trash - Relative Depth", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("Video processing complete.")

# Example usage
video_path = "media/middle.mp4"
process_video_for_depth(video_path)  # Run with a video file

# To use webcam instead:
# process_video_for_depth(0)