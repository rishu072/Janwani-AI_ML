from ultralytics import YOLO

def main():
    # 1. Load the model
    model = YOLO('runs/detect/civic_model/weights/best.pt')

    # 2. Set your image
    source = 'image copy 8.png'  # Replace with your image filename

    # 3. Run Detection for BOTH classes
    # classes=[1, 3, 4, 9] -> Look for Potholes (1, 3, 9) AND Garbage (4)
    # conf=0.15 -> A balanced threshold to catch smaller potholes but avoid fake garbage
    results = model(source, conf=0.15, classes=[1, 3, 4, 9]) 

    # 4. Clean up labels and Save
    for result in results:
        # Rename Pothole classes
        result.names[1] = 'Pothole'
        result.names[3] = 'Pothole'
        result.names[9] = 'Pothole'
        
        # Rename Garbage class
        result.names[4] = 'Garbage'
        
        result.show()
        result.save(filename='combined_result.jpg')
        print("Saved result with Potholes and Garbage!")

if __name__ == '__main__':
    main()