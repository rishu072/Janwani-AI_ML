from ultralytics import YOLO

def main():
    # 1. Load the model
    # "yolo11n.pt" is the Nano model (fastest)
    model = YOLO('yolo11n.pt') 

    # 2. Train the model
    print("Starting training...")
    model.train(
        data='dataset/data.yaml',  # Points to the file you just edited
        epochs=30,                 # 30 runs through the data
        imgsz=640,                 # Resize images to 640px
        batch=4,                   # Low batch size to prevent crashing your laptop
        name='civic_model',        # Results saved in runs/detect/civic_model
        device='cpu'               # Force CPU (since you had DLL errors earlier)
    )
    print("Training Complete!")

if __name__ == '__main__':
    main()