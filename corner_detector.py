import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math

@dataclass
class RoomCornerParams:
    """Basic parameters for room corner detection"""
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 50
    min_line_length: int = 100
    max_line_gap: int = 10
    min_angle: float = 75
    max_angle: float = 105

class RoomCornerDetector:
    def __init__(self, params: Optional[RoomCornerParams] = None):
        self.params = params or RoomCornerParams()

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.params.canny_low, self.params.canny_high)
        
        # Line detection
        lines = cv2.HoughLinesP(edges,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.params.hough_threshold,
                               minLineLength=self.params.min_line_length,
                               maxLineGap=self.params.max_line_gap)
        
        corners = []
        output = frame.copy()
        
        if lines is not None:
            # Draw lines and find corners
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Find intersections
            corners = self._find_corners(lines)
            
            # Draw corners
            for i, (x, y) in enumerate(corners):
                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
        
        return output, corners

    def _find_corners(self, lines) -> List[Tuple[int, int]]:
        corners = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                # Get lines
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # Calculate angles
                angle1 = math.atan2(y2 - y1, x2 - x1)
                angle2 = math.atan2(y4 - y3, x4 - x3)
                
                # Convert to degrees
                angle_diff = abs(math.degrees(angle1 - angle2))
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Check if angle is within range
                if self.params.min_angle <= angle_diff <= self.params.max_angle:
                    # Find intersection
                    corner = self._line_intersection(lines[i][0], lines[j][0])
                    if corner:
                        corners.append(corner)
        
        return self._filter_nearby_corners(corners)

    def _line_intersection(self, line1, line2) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None
            
        # Calculate intersection point
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        
        # Check if point is within frame bounds
        if 0 <= px <= 640 and 0 <= py <= 480:
            return (int(px), int(py))
        return None

    def _filter_nearby_corners(self, corners: List[Tuple[int, int]], min_distance: int = 20) -> List[Tuple[int, int]]:
        if not corners:
            return []
            
        filtered = []
        for corner in corners:
            # Check if corner is too close to any already filtered corner
            if not any(math.sqrt((corner[0]-x)**2 + (corner[1]-y)**2) < min_distance 
                      for x, y in filtered):
                filtered.append(corner)
        return filtered

def main():
    # Initialize detector
    detector = RoomCornerDetector()
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting corner detection. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, corners = detector.detect(frame)
            
            # Display
            cv2.imshow('Room Corner Detection', processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()