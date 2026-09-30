import gradio as gr
import cv2
from ultralytics import YOLO
import os

# Load model
model = YOLO('best.pt')

def process_video(input_video):
    if input_video is None:
        return None, None

    cap = cv2.VideoCapture(input_video)
    
    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback

    # --- SPEED FIX: Resize output to 640x360 for fast processing ---
    out_width, out_height = 640, 360
    
    out_path = 'output_tracking.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # This codec works everywhere
    out = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))
    
    # CSV setup
    csv_path = 'player_positions.csv'
    csv_file = open(csv_path, 'w')
    csv_file.write("frame,player_id,x_center,y_center\n")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- SPEED FIX: Resize each frame before processing ---
        frame = cv2.resize(frame, (out_width, out_height))
        
        # Run YOLO with smaller image size
        results = model.track(frame, persist=True, verbose=False, imgsz=320)
        annotated = results[0].plot()
        
        # Write frame to output video
        out.write(annotated)
        
        # Save positions to CSV
        if results[0].boxes.id is not None:
            for box, id in zip(results[0].boxes.xywh, results[0].boxes.id):
                csv_file.write(f"{frame_idx},{int(id)},{float(box[0])},{float(box[1])}\n")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    csv_file.close()
    
    return out_path, csv_path

# Gradio interface
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Cricket Video"),
    outputs=[
        gr.Video(label="Tracked Video"),
        gr.File(label="Analytics CSV")
    ],
    title="Cricket Player Analytics",
    description="Upload a cricket video to detect and track players.",
    allow_flagging="never"
)

# Launch with correct port for Render
import os
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
