import cv2
import numpy as np

class CornerDetector:
    def __init__(self, max_corners=100, quality_level=0.01, min_distance=10):
        """
        Initialize corner detector with simpler, more reliable parameters.
        
        Args:
            max_corners: Maximum number of corners to detect
            quality_level: Minimum quality of corner below which everyone is rejected
            min_distance: Minimum euclidean distance between corners
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

    def detect_corners(self, frame):
        """
        Detect corners using Shi-Tomasi method (more stable than Harris).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        
        # Process corners if any were found
        if corners is not None:
            # Convert to integer coordinates using int32 instead of int0
            corners = np.int32(corners)
            
            # Draw corners on frame
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green circles
                
            # Add corner count
            cv2.putText(frame, f'Corners: {len(corners)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)  # White text
        else:
            # If no corners found, display message
            cv2.putText(frame, 'No corners detected', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)  # Red text
        
        return frame

    def process_video_feed(self):
        """Process live video feed with corner detection."""
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Starting corner detection...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Process frame
            processed_frame = self.detect_corners(frame)
            
            # Display frame
            cv2.imshow('Corner Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                frame_count += 1
                filename = f'corner_detection_frame_{frame_count}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f"Saved frame as {filename}")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Create detector with default settings
    detector = CornerDetector(
        max_corners=100,       # Detect up to 100 corners
        quality_level=0.1,     # Higher quality level
        min_distance=10        # Minimum 10 pixels between corners
    )
    
    # Start detection
    detector.process_video_feed()

if __name__ == "__main__":
    main()