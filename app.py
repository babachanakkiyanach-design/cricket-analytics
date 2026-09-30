import gradio as gr
import cv2
from ultralytics import YOLO
import os

model = YOLO('best.pt')

def process_video(input_video):
    cap = cv2.VideoCapture(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out_path = 'output_tracking.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    
    with open('player_positions.csv', 'w') as f:
        f.write("frame,player_id,x_center,y_center\n")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)
            
            if results[0].boxes.id is not None:
                for box, id in zip(results[0].boxes.xywh, results[0].boxes.id):
                    f.write(f"{frame_idx},{int(id)},{float(box[0])},{float(box[1])}\n")
            frame_idx += 1
    
    cap.release()
    out.release()
    return out_path, 'player_positions.csv'

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Cricket Video"),
    outputs=[gr.Video(label="Tracked Video"), gr.File(label="Analytics CSV")],
    title="Cricket Player Analytics",
    description="Upload a cricket video to detect and track players."
)

demo.launch()