"""
Object Tracking implementation.
This module provides the ObjectTracker class for tracking objects across frames.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    """Class representing a tracked object."""
    
    def __init__(self, detection, track_id):
        """
        Initialize a new track.
        
        Args:
            detection: Detection in format [x1, y1, x2, y2, confidence, class_id, class_name]
            track_id: Unique identifier for this track
        """
        self.track_id = track_id
        self.bbox = detection[:4]  # [x1, y1, x2, y2]
        self.confidence = detection[4]
        self.class_id = detection[5]
        self.class_name = detection[6]
        
        # Track state
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = 'confirmed'  # 'tentative', 'confirmed', 'deleted'
        
        # Motion estimation (simple)
        self.bbox_history = [self.bbox]
        self.vx = 0  # X velocity
        self.vy = 0  # Y velocity
    
    def update(self, detection):
        """Update track with new detection."""
        self.bbox = detection[:4]
        self.confidence = detection[4]
        
        # Update motion model
        if len(self.bbox_history) > 0:
            prev_bbox = self.bbox_history[-1]
            cx_prev, cy_prev = (prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2
            cx, cy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
            self.vx = cx - cx_prev
            self.vy = cy - cy_prev
        
        self.bbox_history.append(self.bbox)
        if len(self.bbox_history) > 30:  # Limit history
            self.bbox_history.pop(0)
        
        self.hits += 1
        self.time_since_update = 0
        self.state = 'confirmed'
    
    def predict(self):
        """Predict next position using motion model."""
        if len(self.bbox_history) > 0:
            # Simple velocity model
            x1, y1, x2, y2 = self.bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Apply velocity
            cx += self.vx
            cy += self.vy
            
            # Convert back to bbox
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            self.bbox = [x1, y1, x2, y2]
        
        self.age += 1
        self.time_since_update += 1
        
        return self.bbox
    
    def get_state(self):
        """Get current tracking info in the detection format."""
        return [*self.bbox, self.confidence, self.class_id, self.class_name, self.track_id]


class ObjectTracker:
    """
    Multiple object tracker that associates detections with existing tracks.
    This implementation provides a simplified version of popular trackers like ByteTrack.
    """
    
    def __init__(self, tracker_type='bytetrack', max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the tracker.
        
        Args:
            tracker_type: Type of tracking algorithm to use
            max_age: Maximum number of frames to keep lost tracks
            min_hits: Minimum number of detections before track is confirmed
            iou_threshold: IOU threshold for detection association
        """
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IOU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _associate_detections_to_tracks(self, detections):
        """Associate detections with existing tracks using IoU."""
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate IoU for each detection-track pair
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d_idx, detection in enumerate(detections):
            for t_idx, track in enumerate(self.tracks):
                iou_matrix[d_idx, t_idx] = self._calculate_iou(detection[:4], track.bbox)
        
        # Use Hungarian algorithm for optimal assignment
        detection_indices, track_indices = linear_sum_assignment(-iou_matrix)
        
        # Create lists for matched, unmatched detections and tracks
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Filter matches by IoU threshold
        for d_idx, t_idx in zip(detection_indices, track_indices):
            if iou_matrix[d_idx, t_idx] >= self.iou_threshold:
                matches.append((d_idx, t_idx))
                unmatched_detections.remove(d_idx)
                unmatched_tracks.remove(t_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections in format [x1, y1, x2, y2, confidence, class_id, class_name]
            
        Returns:
            List of active tracks with their current state
        """
        # 1. Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # 2. Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # 3. Update matched tracks
        for d_idx, t_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])
        
        # 4. Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            self.tracks.append(Track(detections[d_idx], self.track_id_count))
            self.track_id_count += 1
        
        # 5. Delete old tracks
        new_tracks = []
        for i, track in enumerate(self.tracks):
            # Keep track if:
            # - Recently updated OR
            # - Confirmed and not too old
            if track.time_since_update == 0 or (
                track.hits >= self.min_hits and 
                track.time_since_update <= self.max_age
            ):
                new_tracks.append(track)
        
        self.tracks = new_tracks
        
        # 6. Return active tracks (all tracks that are not too old)
        results = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                results.append(track.get_state())
        
        return results
