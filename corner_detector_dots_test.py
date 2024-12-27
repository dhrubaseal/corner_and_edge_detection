import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass
class WallDetectionParams:
    """Parameters for wall corner detection"""
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 50
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    corner_neighborhood: int = 20
    corner_confidence: float = 0.8

class WallCornerDetector:
    def __init__(self, params: Optional[WallDetectionParams] = None):
        self.params = params or WallDetectionParams()

    def detect_wall_corners(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Detect corners of walls using edge detection and line intersection.
        
        Returns:
            Tuple of (processed frame, corner coordinates)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to enhance walls
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        
        # Edge detection optimized for walls
        edges = cv2.Canny(equalized, 
                         self.params.canny_low, 
                         self.params.canny_high)
        
        # Dilate edges to connect gaps
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(dilated,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.params.hough_threshold,
                               minLineLength=self.params.hough_min_line_length,
                               maxLineGap=self.params.hough_max_line_gap)
        
        corners = []
        if lines is not None:
            # Find line intersections
            corners = self._find_line_intersections(lines)
            
            # Filter corners based on edge strength
            corners = self._filter_corners(corners, edges)
            
            # Non-maximum suppression to remove duplicate corners
            corners = self._non_max_suppression(corners)
        
        return self._visualize_results(frame, lines, corners)

    def _find_line_intersections(self, lines: np.ndarray) -> List[Tuple[int, int]]:
        """Find intersection points between detected lines"""
        corners = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # Calculate intersection
                den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if den != 0:  # Lines are not parallel
                    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
                    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
                    
                    # Check if intersection is within image bounds
                    if 0 <= px <= 640 and 0 <= py <= 480:
                        # Check if intersection is near line endpoints
                        if self._is_near_endpoints((px, py), lines[i][0], lines[j][0]):
                            corners.append((int(px), int(py)))
        
        return corners

    def _is_near_endpoints(self, 
                          intersection: Tuple[float, float], 
                          line1: np.ndarray, 
                          line2: np.ndarray,
                          threshold: float = 10.0) -> bool:
        """Check if intersection point is near the endpoints of both lines"""
        x, y = intersection
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate distances to line endpoints
        d1 = min(np.sqrt((x-x1)**2 + (y-y1)**2),
                 np.sqrt((x-x2)**2 + (y-y2)**2))
        d2 = min(np.sqrt((x-x3)**2 + (y-y3)**2),
                 np.sqrt((x-x4)**2 + (y-y4)**2))
        
        return d1 < threshold and d2 < threshold

    def _filter_corners(self, 
                       corners: List[Tuple[int, int]], 
                       edges: np.ndarray) -> List[Tuple[int, int]]:
        """Filter corners based on edge strength in neighborhood"""
        filtered_corners = []
        
        for x, y in corners:
            # Extract neighborhood around corner
            y1 = max(0, y - self.params.corner_neighborhood//2)
            y2 = min(edges.shape[0], y + self.params.corner_neighborhood//2)
            x1 = max(0, x - self.params.corner_neighborhood//2)
            x2 = min(edges.shape[1], x + self.params.corner_neighborhood//2)
            
            neighborhood = edges[y1:y2, x1:x2]
            
            # Calculate edge density in neighborhood
            edge_density = np.sum(neighborhood > 0) / neighborhood.size
            
            if edge_density > self.params.corner_confidence:
                filtered_corners.append((x, y))
        
        return filtered_corners

    def _non_max_suppression(self, 
                            corners: List[Tuple[int, int]], 
                            distance_threshold: int = 20) -> List[Tuple[int, int]]:
        """Remove duplicate corners that are too close to each other"""
        if not corners:
            return []
            
        corners = np.array(corners)
        selected = np.zeros(len(corners), dtype=bool)
        
        # Calculate pairwise distances
        distances = np.sqrt(((corners[:, None] - corners) ** 2).sum(axis=2))
        
        # Sort corners by number of neighbors
        neighbor_count = (distances < distance_threshold).sum(axis=1)
        corner_order = np.argsort(-neighbor_count)
        
        for idx in corner_order:
            if not selected[idx]:
                # Select this corner
                selected[idx] = True
                
                # Suppress neighbors
                neighbors = distances[idx] < distance_threshold
                selected[neighbors] = False
                selected[idx] = True
                
        return [tuple(corner) for corner, select in zip(corners, selected) if select]

    def _visualize_results(self, 
                          frame: np.ndarray, 
                          lines: Optional[np.ndarray],
                          corners: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Visualize detected lines and corners"""
        viz_frame = frame.copy()
        
        # Draw detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Draw corners
        for i, (x, y) in enumerate(corners):
            cv2.circle(viz_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(viz_frame, str(i), (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add statistics
        cv2.putText(viz_frame, f'Corners: {len(corners)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return viz_frame, corners

    def process_video_feed(self, source: int = 0) -> None:
        """Process video feed with wall corner detection"""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open video source")

        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting wall corner detection...")
        print("Controls:")
        print("- 'q': Quit")
        print("- 's': Save frame")
        print("- '+'/'-': Adjust edge detection threshold")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, corners = self.detect_wall_corners(frame)
                
                # Display frame
                cv2.imshow('Wall Corner Detection', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'wall_corners_{frame_count}.jpg'
                    cv2.imwrite(filename, processed_frame)
                    print(f"Saved frame as {filename}")
                    frame_count += 1
                elif key == ord('+'):
                    self.params.canny_high = min(255, self.params.canny_high + 10)
                elif key == ord('-'):
                    self.params.canny_high = max(0, self.params.canny_high - 10)
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    # Create detector with custom parameters
    params = WallDetectionParams(
        canny_low=50,
        canny_high=150,
        hough_threshold=50,
        hough_min_line_length=50,
        hough_max_line_gap=10,
        corner_neighborhood=20,
        corner_confidence=0.8
    )
    
    detector = WallCornerDetector(params)
    detector.process_video_feed()

if __name__ == "__main__":
    main()