from ultralytics import YOLO

def main():
    # 1. Load your trained model
    model = YOLO('runs/detect/civic_model/weights/best.pt')

    # 2. Define your test image
    # Change 'garbage_image.jpg' to the actual name of your photo
    source = 'image copy .png' 

    # 3. Run Prediction
    # classes=[4] tells it to ONLY look for Garbage. 
    # conf=0.25 is the standard threshold (ignore weak guesses).
    results = model(source, conf=0.25, classes=[4]) 

    # 4. Show and Save
    for result in results:
        # Rename the label so it looks clean
        result.names[4] = 'Garbage'
        
        result.show()
        result.save(filename='garbage_test_result.jpg')
        print("Saved garbage detection result!")

if __name__ == '__main__':
    main()