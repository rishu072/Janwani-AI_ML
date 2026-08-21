from ultralytics import YOLO

def main():
    model = YOLO('runs/detect/civic_model/weights/best.pt')

    # THE FIX: Add 'classes=[1, 3, 9]'
    # This tells the model: "Only look for Class 1, 3, and 9 (Potholes)."
    # It will completely BLIND itself to Class 4 (Garbage), so the human won't be detected.
    results = model('image copy 4.png', conf=0.1, classes=[1, 3, 9]) 

    for result in results:
        # Renaming for display
        result.names[1] = 'Pothole'
        result.names[3] = 'Pothole'
        result.names[9] = 'Pothole'
        
        result.show()
        result.save(filename='pothole_filtered.jpg')
        print("Saved! The human/garbage should be gone now.")

if __name__ == '__main__':
    main()