from ultralytics import YOLO

# Load the checkpoint from your screenshot
model = YOLO('runs/detect/civic_model/weights/last.pt')

# Resume training where it left off
model.train(resume=True)