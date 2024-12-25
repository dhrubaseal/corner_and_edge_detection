import cv2
import numpy as np

class LineDetector:
    def __init__(self, 
                 min_length=100,    # Minimum line length
                 max_gap=10,        # Maximum gap between line segments
                 threshold=50):     # Accumulator threshold
        """
        Initialize line detector with parameters.
        
        Args:
            min_length: Minimum length of line
            max_gap: Maximum gap allowed between line segments
            threshold: Accumulator threshold for Hough lines
        """
        self.min_length = min_length
        self.max_gap = max_gap
        self.threshold = threshold

    def detect_lines(self, frame):
        """
        Detect lines in the frame using Hough Line Transform.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.threshold,
            minLineLength=self.min_length,
            maxLineGap=self.max_gap
        )
        
        # Create a copy for visualization
        line_image = frame.copy()
        
        # Draw lines if any were found
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add line count
            cv2.putText(line_image, f'Lines: {len(lines)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
        else:
            # If no lines found, display message
            cv2.putText(line_image, 'No lines detected', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        return line_image, edges  # Return both processed image and edge detection

    def process_video_feed(self):
        """Process live video feed with line detection."""
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Starting line detection...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'e' to toggle edge view")
        
        show_edges = False
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Process frame
            line_image, edges = self.detect_lines(frame)
            
            # Display frame (either lines or edges based on toggle)
            if show_edges:
                cv2.imshow('Line Detection', edges)
            else:
                cv2.imshow('Line Detection', line_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                frame_count += 1
                filename = f'line_detection_frame_{frame_count}.jpg'
                cv2.imwrite(filename, line_image)
                print(f"Saved frame as {filename}")
            elif key == ord('e'):
                # Toggle edge view
                show_edges = not show_edges
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Create detector with default settings
    detector = LineDetector(
        min_length=100,    # Minimum line length in pixels
        max_gap=10,        # Maximum gap between line segments
        threshold=50       # Accumulator threshold
    )
    
    # Start detection
    detector.process_video_feed()

if __name__ == "__main__":
    main()