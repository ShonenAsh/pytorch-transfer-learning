# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# Extension: Live feed MNIST Digit classification

import cv2
import torch
import sys
from model import MyNetwork

# Apply preprocessing pipeline (described in the report)
def preprocess_frame(roi):
    # Convert grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
    resized_roi = cv2.resize(gray_roi, (28, 28)) # resize
    _, inv_roi = cv2.threshold(cv2.bitwise_not(resized_roi), 127, 255, cv2.THRESH_OTSU)
    # normalized_roi = inv_roi / 255.0
    
    # obtain a tensor
    tensor_roi = torch.from_numpy(inv_roi).float().unsqueeze(0).unsqueeze(0)
    
    return tensor_roi, inv_roi

# accepts a model file as the only argument
def main(argv):
    if len(argv) != 2:
        print("Usage: python extension.py <path_to_model>")
        sys.exit(1)

    model_path = sys.argv[1]
    
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    cap = cv2.VideoCapture(0)
    # this is necessary to make it work on WSL
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera failed!")
            break

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # get the top left corner
        top_left_x = center_x - 140
        top_left_y = center_y - 140
        
        # handle edge case - ensure the corner is within the frame
        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        
        # extract the 280x280 region of interest (ROI)
        roi = frame[top_left_y:top_left_y+280, top_left_x:top_left_x+280]
        
        # check if roi valid (not empty and has the right dimensions)
        if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            # preprocessed  ROI
            input_tensor, mnist_sized_roi = preprocess_frame(roi) 
            with torch.no_grad():
                output = model(input_tensor)
            
            _, predicted = torch.max(output, 1)
            label = predicted.item()
            
            # draw a 280x280 square on the centered frame (ROI)
            cv2.rectangle(frame, (top_left_x, top_left_y), 

                         (top_left_x+280, top_left_y+280), (0, 255, 0), 2)

            
            # add prediction as Text

            cv2.putText(frame, f"Predicted: {label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # slice a display from the frame and set it to the processed ROI
            # display the 28x28 MNIST-sized image in the corner

            display_size = 112  # 4x28 (resized) 
            mnist_display = cv2.resize(mnist_sized_roi, (display_size, display_size), 
                                     interpolation=cv2.INTER_NEAREST)
            
            frame[10:10+display_size, width-10-display_size:width-10] = cv2.cvtColor(mnist_display, cv2.COLOR_GRAY2BGR)
            
            # Draw a border around the MNIST display
            cv2.rectangle(frame, (width-10-display_size, 10), 
                         (width-10, 10+display_size), (255, 0, 0), 2)
        else:
            cv2.putText(frame, f"Out of frame", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        # display the feed
        cv2.imshow('Live Camera Feed', frame)

        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# accepts a model pth file as the only argument

if __name__ == "__main__":
    main(sys.argv)

