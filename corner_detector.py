import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import math
from collections import defaultdict

@dataclass
class RoomCornerParams:
    """Parameters for room corner detection with noise reduction"""
    # Edge detection parameters
    canny_low: int = 80  # Increased from 50
    canny_high: int = 150  # Increased from 150
    blur_kernel_size: Tuple[int, int] = (7, 7)  # Increased blur for noise reduction
    
    # Line detection parameters
    hough_threshold: int = 60  # Increased from 50
    min_line_length: int = 100  # Increased from 100
    max_line_gap: int = 5  # Reduced from 10
    
    # Corner detection parameters
    min_angle: float = 80  # More strict angle
    max_angle: float = 100  # More strict angle
    min_corner_distance: int = 30  # Increased from 20
    
    # Frame parameters
    frame_width: int = 640
    frame_height: int = 480
    
    # Advanced parameters
    line_cluster_threshold: float = 0.1  # Reduced for stricter clustering
    min_line_strength: int = 50  # Increased minimum line strength
    temporal_persistence: int = 5  # Frames a corner must persist
    confidence_threshold: float = 3.0  # Minimum confidence for corner display

class RoomCornerDetector:
    def __init__(self, params: Optional[RoomCornerParams] = None):
        self.params = params or RoomCornerParams()
        self._corner_history: Dict[Tuple[int, int], float] = defaultdict(float)
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        self._frame_count += 1
        
        # Ensure frame is properly sized
        frame = cv2.resize(frame, (self.params.frame_width, self.params.frame_height))
        
        # Enhanced preprocessing with noise reduction
        preprocessed = self._preprocess_frame(frame)
        
        # Improved line detection
        lines = self._detect_lines(preprocessed)
        
        output = frame.copy()
        corners = []
        
        if lines is not None:
            # Cluster and filter lines
            clustered_lines = self._cluster_lines(lines)
            filtered_clusters = self._filter_line_clusters(clustered_lines)
            
            # Draw only the strongest lines
            for cluster in filtered_clusters:
                if cluster:
                    strongest_line = max(cluster, key=lambda x: x[1])[0]
                    x1, y1, x2, y2 = strongest_line
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Find corners with strict filtering
            current_corners = self._find_corners(filtered_clusters)
            corners = self._apply_temporal_smoothing(current_corners)
            
            # Draw only high-confidence corners
            self._draw_corners(output, corners)
        
        return output, corners

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with stronger noise reduction"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply stronger bilateral filter for edge preservation while reducing noise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply larger kernel Gaussian blur
        blurred = cv2.GaussianBlur(denoised, self.params.blur_kernel_size, 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Enhanced Canny edge detection
        edges = cv2.Canny(thresh, self.params.canny_low, self.params.canny_high)
        
        # Morphological operations to clean up edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges

    def _detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """Line detection with stricter filtering"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.params.hough_threshold,
            minLineLength=self.params.min_line_length,
            maxLineGap=self.params.max_line_gap
        )
        
        if lines is None:
            return None
        
        # Filter weak and non-vertical/horizontal lines
        strong_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1))) % 180
            
            # Keep only strong lines that are roughly vertical or horizontal
            is_vertical = abs(angle - 90) < 20
            is_horizontal = angle < 20 or angle > 160
            
            if (length >= self.params.min_line_length and 
                (is_vertical or is_horizontal)):
                strong_lines.append(line)
        
        return np.array(strong_lines) if strong_lines else None

    def _cluster_lines(self, lines: np.ndarray) -> List[List[Tuple[np.ndarray, float]]]:
        """Cluster similar lines together to reduce noise"""
        clusters = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2-y1, x2-x1)
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Try to add to existing cluster
            added = False
            for cluster in clusters:
                ref_line = cluster[0][0]
                ref_angle = math.atan2(ref_line[3]-ref_line[1], 
                                     ref_line[2]-ref_line[0])
                
                # Calculate minimum angle difference
                angle_diff = abs(angle - ref_angle)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                
                if angle_diff < self.params.line_cluster_threshold:
                    cluster.append((line[0], length))
                    added = True
                    break
            
            # Create new cluster if not added to existing ones
            if not added:
                clusters.append([(line[0], length)])
        
        return clusters

    def _filter_line_clusters(self, clusters: List[List[Tuple[np.ndarray, float]]]) -> List[List[Tuple[np.ndarray, float]]]:
        """Filter line clusters to reduce noise"""
        filtered_clusters = []
        
        for cluster in clusters:
            if len(cluster) >= 2:  # Only keep clusters with multiple supporting lines
                # Calculate average angle for cluster
                angles = [math.atan2(line[0][3]-line[0][1], line[0][2]-line[0][0]) 
                         for line in cluster]
                mean_angle = np.mean(angles)
                
                # Keep only lines close to mean angle
                filtered_lines = [
                    line for line in cluster
                    if abs(math.atan2(line[0][3]-line[0][1], line[0][2]-line[0][0]) - mean_angle) < 0.1
                ]
                
                if filtered_lines:
                    filtered_clusters.append(filtered_lines)
        
        return filtered_clusters

    def _find_corners(self, clustered_lines: List[List[Tuple[np.ndarray, float]]]) -> List[Tuple[int, int]]:
        """Find corners with stricter angle requirements"""
        corners = []
        
        # Use only the strongest line from each cluster
        strong_lines = [max(cluster, key=lambda x: x[1])[0] for cluster in clustered_lines]
        
        for i in range(len(strong_lines)):
            for j in range(i + 1, len(strong_lines)):
                line1 = strong_lines[i]
                line2 = strong_lines[j]
                
                # Calculate angle between lines
                angle1 = math.atan2(line1[3]-line1[1], line1[2]-line1[0])
                angle2 = math.atan2(line2[3]-line2[1], line2[2]-line2[0])
                angle_diff = abs(math.degrees(angle1 - angle2))
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Stricter angle checking
                if (self.params.min_angle <= angle_diff <= self.params.max_angle):
                    corner = self._line_intersection(line1, line2)
                    if corner:
                        corners.append(corner)
        
        return self._filter_nearby_corners(corners)

    def _line_intersection(self, line1, line2) -> Optional[Tuple[int, int]]:
        """Calculate line intersection with bounds checking"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate denominator
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-5:  # Avoid division by very small numbers
            return None
            
        # Calculate intersection point
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        
        # Check if point is within frame bounds with margin
        margin = 10
        if (margin <= px <= self.params.frame_width - margin and 
            margin <= py <= self.params.frame_height - margin):
            return (int(px), int(py))
        return None

    def _filter_nearby_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter corners that are too close to each other"""
        if not corners:
            return []
        
        filtered = []
        for corner in corners:
            if not any(math.sqrt((corner[0]-x)**2 + (corner[1]-y)**2) < 
                      self.params.min_corner_distance for x, y in filtered):
                filtered.append(corner)
        return filtered

    def _apply_temporal_smoothing(self, current_corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Enhanced temporal smoothing with stricter persistence"""
        # Update corner history with decay
        for corner in list(self._corner_history.keys()):
            # Find closest current corner
            closest_current = None
            min_dist = float('inf')
            
            for curr in current_corners:
                dist = math.sqrt((corner[0]-curr[0])**2 + (corner[1]-curr[1])**2)
                if dist < min_dist and dist < self.params.min_corner_distance:
                    min_dist = dist
                    closest_current = curr
            
            if closest_current:
                # Update position with moving average
                self._corner_history[corner] += 1
            else:
                # Decay confidence for unseen corners
                self._corner_history[corner] -= 0.5
            
            # Remove low confidence corners
            if self._corner_history[corner] <= 0:
                del self._corner_history[corner]
        
        # Add new corners
        for corner in current_corners:
            if corner not in self._corner_history:
                self._corner_history[corner] = 1
        
        # Return only high-confidence corners
        stable_corners = [
            corner for corner, conf in self._corner_history.items()
            if conf >= self.params.confidence_threshold
        ]
        
        return stable_corners

    def _draw_corners(self, output: np.ndarray, corners: List[Tuple[int, int]]):
        """Draw corners with confidence-based visualization"""
        for corner in corners:
            confidence = self._corner_history[corner]
            if confidence >= self.params.confidence_threshold:
                # Scale radius and color based on confidence
                radius = min(int(confidence * 1.5), 8)
                color_intensity = min(int(confidence * 30), 255)
                color = (0, color_intensity, 255)
                
                cv2.circle(output, corner, radius, color, -1)
                cv2.circle(output, corner, radius + 2, color, 1)

def main():
    # Initialize detector with stricter parameters
    params = RoomCornerParams(
        canny_low=100,
        canny_high=200,
        hough_threshold=80,
        min_line_length=120,
        max_line_gap=5,
        min_corner_distance=30,
        temporal_persistence=5,
        confidence_threshold=3.0
    )
    detector = RoomCornerDetector(params)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, params.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params.frame_height)
    
    print("Starting corner detection. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, corners = detector.detect(frame)
            
            # Add corner count and confidence display
            cv2.putText(
                processed_frame,
                f"Stable Corners: {len(corners)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            cv2.imshow('Room Corner Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()