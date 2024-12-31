# [Previous imports remain the same]

@dataclass
class RoomCornerParams:
    """Parameters readjusted for better detection"""
    # Edge detection parameters
    canny_low: int = 20  # Lowered significantly for subtle edges
    canny_high: int = 60  # Lowered for more edge detection
    blur_kernel_size: Tuple[int, int] = (5, 5)  # Reduced blur
    
    # Line detection parameters
    hough_threshold: int = 25  # Lowered for better line detection
    min_line_length: int = 60  # Reduced to detect shorter lines
    max_line_gap: int = 15  # Increased to connect broken lines
    
    # Corner detection parameters
    min_angle: float = 70  # More lenient angle
    max_angle: float = 110  # More lenient angle
    min_corner_distance: int = 20  # Reduced to detect closer corners
    
    # Frame parameters
    frame_width: int = 640
    frame_height: int = 480
    
    # Advanced parameters
    line_cluster_threshold: float = 0.2  # More lenient clustering
    min_line_strength: int = 25  # Lower strength requirement
    temporal_persistence: int = 3  # Reduced for faster response
    confidence_threshold: float = 2.0  # Lower confidence requirement

def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
    """Modified preprocessing for better edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Lighter denoising
    denoised = cv2.bilateralFilter(enhanced, 5, 35, 35)
    
    # Lighter blur
    blurred = cv2.GaussianBlur(denoised, self.params.blur_kernel_size, 0)
    
    # Modified thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 7, 2
    )
    
    # Edge detection with lighter morphological operations
    edges = cv2.Canny(thresh, self.params.canny_low, self.params.canny_high)
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

def _detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
    """Modified line detection with more lenient filtering"""
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
    
    strong_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = abs(math.degrees(math.atan2(y2-y1, x2-x1))) % 180
        
        # More lenient angle constraints
        is_vertical = abs(angle - 90) < 25
        is_horizontal = angle < 25 or angle > 155
        
        if (length >= self.params.min_line_length and 
            (is_vertical or is_horizontal)):
            strong_lines.append(line)
    
    return np.array(strong_lines) if strong_lines else None

def _apply_temporal_smoothing(self, current_corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Modified temporal smoothing with faster response"""
    for corner in list(self._corner_history.keys()):
        closest_current = None
        min_dist = float('inf')
        
        for curr in current_corners:
            dist = math.sqrt((corner[0]-curr[0])**2 + (corner[1]-curr[1])**2)
            if dist < min_dist and dist < self.params.min_corner_distance:
                min_dist = dist
                closest_current = curr
        
        if closest_current:
            self._corner_history[corner] += 1.2  # Faster confidence buildup
        else:
            self._corner_history[corner] -= 0.4  # Slower decay
        
        if self._corner_history[corner] <= 0:
            del self._corner_history[corner]
    
    for corner in current_corners:
        if corner not in self._corner_history:
            self._corner_history[corner] = 1.5  # Higher initial confidence
    
    stable_corners = [
        corner for corner, conf in self._corner_history.items()
        if conf >= self.params.confidence_threshold
    ]
    
    return stable_corners

# [Rest of the class implementation remains the same]

def main():
    # Initialize with adjusted parameters
    params = RoomCornerParams()  # Using the adjusted default parameters
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
            
            # Add debug windows
            processed_frame, corners = detector.detect(frame)
            
            # Show the edge detection result
            edges = detector._preprocess_frame(frame)
            cv2.imshow('Edge Detection', edges)
            
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