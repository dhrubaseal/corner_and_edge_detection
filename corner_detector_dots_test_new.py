import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math

@dataclass
class RoomCornerParams:
    """Parameters specifically tuned for room corner detection"""
    # Edge detection
    canny_low: int = 30
    canny_high: int = 100
    
    # Line detection
    hough_threshold: int = 100
    hough_min_line_length: int = 100  # Increased to detect only long wall lines
    hough_max_line_gap: int = 20
    
    # Angle filtering
    min_angle: float = 75  # Minimum angle between walls
    max_angle: float = 105  # Maximum angle between walls
    
    # Corner filtering
    min_corner_distance: int = 50  # Minimum distance between corners
    edge_density_threshold: float = 0.3  # Required edge density around corner

class RoomCornerDetector:
    def __init__(self, params: Optional[RoomCornerParams] = None):
        self.params = params or RoomCornerParams()

    def detect_room_corners(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Detect room corners where walls meet.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Edge detection
        edges = cv2.Canny(denoised, 
                         self.params.canny_low, 
                         self.params.canny_high)
        
        # Dilate edges
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Detect long lines
        lines = cv2.HoughLinesP(dilated,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.params.hough_threshold,
                               minLineLength=self.params.hough_min_line_length,
                               maxLineGap=self.params.hough_max_line_gap)
        
        corners = []
        if lines is not None:
            # Convert lines to angle-based format
            wall_lines = self._process_lines(lines)
            
            # Find corners between walls with appropriate angles
            corners = self._find_wall_corners(wall_lines)
            
            # Filter corners based on room geometry
            corners = self._filter_room_corners(corners, edges)
        
        return self._visualize_results(frame, wall_lines if lines is not None else None, corners)

    def _process_lines(self, lines):
        """Convert lines to angle-based format and merge similar lines"""
        processed_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            # Normalize angle to 0-180
            if angle < 0:
                angle += 180
            
            # Group similar angles (within 10 degrees)
            merged = False
            for existing in processed_lines:
                if abs(existing['angle'] - angle) < 10 or abs(abs(existing['angle'] - angle) - 180) < 10:
                    merged = True
                    break
            
            if not merged:
                processed_lines.append({
                    'points': (x1, y1, x2, y2),
                    'angle': angle
                })
        
        return processed_lines

    def _find_wall_corners(self, wall_lines):
        """Find corners between walls with appropriate angles"""
        corners = []
        
        for i in range(len(wall_lines)):
            for j in range(i + 1, len(wall_lines)):
                # Calculate angle between lines
                angle_diff = abs(wall_lines[i]['angle'] - wall_lines[j]['angle'])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Check if angle is appropriate for room corners
                if self.params.min_angle <= angle_diff <= self.params.max_angle:
                    intersection = self._line_intersection(
                        wall_lines[i]['points'],
                        wall_lines[j]['points']
                    )
                    if intersection:
                        corners.append(intersection)
        
        return corners

    def _line_intersection(self, line1, line2):
        """Calculate intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None
            
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denominator
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denominator
        
        # Check if intersection is within frame bounds
        if 0 <= px <= 640 and 0 <= py <= 480:
            return (int(px), int(py))
        return None

    def _filter_room_corners(self, corners, edges):
        """Filter corners based on room geometry and edge density"""
        filtered_corners = []
        
        for corner in corners:
            # Check edge density around corner
            x, y = corner
            region = edges[
                max(0, y-10):min(edges.shape[0], y+10),
                max(0, x-10):min(edges.shape[1], x+10)
            ]
            edge_density = np.sum(region > 0) / region.size
            
            if edge_density >= self.params.edge_density_threshold:
                # Check distance from other corners
                if not any(self._distance(corner, existing) < self.params.min_corner_distance 
                         for existing in filtered_corners):
                    filtered_corners.append(corner)
        
        return filtered_corners

    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _visualize_results(self, frame, wall_lines, corners):
        """Visualize detected walls and corners"""
        viz_frame = frame.copy()
        
        # Draw detected wall lines
        if wall_lines:
            for line in wall_lines:
                x1, y1, x2, y2 = line['points']
                cv2.line(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw corners
        for i, (x, y) in enumerate(corners):
            cv2.circle(viz_frame, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(viz_frame, str(i), (x+15, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return viz_frame, corners

    def process_video_feed(self, source: int = 0):
        """Process video feed with room corner detection"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open video source")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting room corner detection...")
        print("Controls:")
        print("- 'q': Quit")
        print("- 's': Save frame")
        print("- '+'/'-': Adjust line length threshold")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, corners = self.detect_room_corners(frame)
                cv2.imshow('Room Corner Detection', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('room_corners.jpg', processed_frame)
                    print("Saved frame")
                elif key == ord('+'):
                    self.params.hough_min_line_length += 10
                elif key == ord('-'):
                    self.params.hough_min_line_length = max(50, 
                        self.params.hough_min_line_length - 10)
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    params = RoomCornerParams(
        canny_low=30,
        canny_high=100,
        hough_threshold=100,
        hough_min_line_length=100,
        hough_max_line_gap=20,
        min_angle=75,
        max_angle=105,
        min_corner_distance=50,
        edge_density_threshold=0.3
    )
    
    detector = RoomCornerDetector(params)
    detector.process_video_feed()

if __name__ == "__main__":
    main()