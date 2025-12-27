"""
Feature Extractor

Centralized behavioral feature extraction for anomaly detection.
Extracts 10+ features including sequence entropy, file sensitivity, and velocity vectors.
"""

import hashlib
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from observability.logging_config import get_logger

logger = get_logger(__name__)


# File sensitivity patterns
SENSITIVITY_PATTERNS = {
    "critical": [
        r"\.env($|\.)",
        r"secrets?/",
        r"credentials?/",
        r"\.pem$",
        r"\.key$",
        r"\.p12$",
        r"id_rsa",
        r"password",
        r"api_key",
        r"token",
    ],
    "high": [
        r"config/",
        r"settings\.py",
        r"auth/",
        r"\.git/config",
        r"\.ssh/",
        r"\.npmrc",
        r"\.pypirc",
    ],
    "medium": [
        r"docker-compose",
        r"Dockerfile",
        r"\.yaml$",
        r"\.yml$",
        r"\.toml$",
        r"database/",
    ],
    "low": [
        r"\.py$",
        r"\.js$",
        r"\.ts$",
        r"\.md$",
        r"\.txt$",
    ],
}

# Compile patterns
COMPILED_PATTERNS = {
    level: [re.compile(p, re.I) for p in patterns]
    for level, patterns in SENSITIVITY_PATTERNS.items()
}


@dataclass
class FeatureVector:
    """Complete feature vector for anomaly detection."""
    
    # Time-based features
    time_since_last: float  # Seconds since last request
    unusual_hour: int  # 1 if night/weekend, 0 otherwise
    session_duration: float  # Seconds since session start
    
    # Rate features
    request_rate_per_min: float  # Requests in last minute
    velocity_change: float  # Acceleration (rate change rate)
    
    # Tool pattern features
    tool_changed: int  # 1 if different tool from last
    sequence_entropy: float  # Tool sequence randomness
    tool_transition_prob: float  # Markov transition probability
    
    # Access pattern features
    file_sensitivity_score: float  # 0-1 based on file paths
    resource_depth_score: float  # Directory depth normalized
    new_ip: int  # 1 if unseen IP
    
    # Request features
    request_size_anomaly: float  # Z-score of request size
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.time_since_last,
            self.unusual_hour,
            self.session_duration,
            self.request_rate_per_min,
            self.velocity_change,
            self.tool_changed,
            self.sequence_entropy,
            self.tool_transition_prob,
            self.file_sensitivity_score,
            self.resource_depth_score,
            self.new_ip,
            self.request_size_anomaly,
        ])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "time_since_last": self.time_since_last,
            "unusual_hour": float(self.unusual_hour),
            "session_duration": self.session_duration,
            "request_rate_per_min": self.request_rate_per_min,
            "velocity_change": self.velocity_change,
            "tool_changed": float(self.tool_changed),
            "sequence_entropy": self.sequence_entropy,
            "tool_transition_prob": self.tool_transition_prob,
            "file_sensitivity_score": self.file_sensitivity_score,
            "resource_depth_score": self.resource_depth_score,
            "new_ip": float(self.new_ip),
            "request_size_anomaly": self.request_size_anomaly,
        }
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered feature names."""
        return [
            "time_since_last",
            "unusual_hour",
            "session_duration",
            "request_rate_per_min",
            "velocity_change",
            "tool_changed",
            "sequence_entropy",
            "tool_transition_prob",
            "file_sensitivity_score",
            "resource_depth_score",
            "new_ip",
            "request_size_anomaly",
        ]


class BehavioralFeatureExtractor:
    """
    Extract behavioral features for anomaly detection.
    
    Maintains per-user state for velocity calculations and pattern tracking.
    """
    
    # Tool transition probabilities (learned from normal traffic)
    DEFAULT_TRANSITIONS = {
        ("read_file", "read_file"): 0.4,
        ("read_file", "search_files"): 0.3,
        ("search_files", "read_file"): 0.5,
        ("git_status", "git_diff"): 0.6,
        ("git_diff", "read_file"): 0.4,
        ("list_dir", "read_file"): 0.5,
    }
    
    def __init__(self, window_size: int = 100):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Size of sliding window for pattern analysis
        """
        self.window_size = window_size
        
        # Per-user state
        self._user_sessions: Dict[str, datetime] = {}  # session start times
        self._user_ips: Dict[str, set] = {}  # known IPs per user
        self._user_tool_history: Dict[str, deque] = {}  # recent tools
        self._user_rate_history: Dict[str, deque] = {}  # request rates
        self._request_sizes: deque = deque(maxlen=1000)  # global request sizes
        
        # Transition probabilities (will be updated from data)
        self.transitions = dict(self.DEFAULT_TRANSITIONS)
    
    def extract_features(
        self,
        tool_name: str,
        user_id: str,
        timestamp: datetime,
        ip: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> FeatureVector:
        """
        Extract all behavioral features for a request.
        
        Args:
            tool_name: Name of tool being called
            user_id: User identifier
            timestamp: Request timestamp
            ip: Optional IP address
            params: Optional tool parameters
            history: Optional list of recent requests for this user
            
        Returns:
            FeatureVector with all extracted features
        """
        history = history or []
        params = params or {}
        
        # Initialize user state if needed
        self._init_user_state(user_id, timestamp)
        
        # Extract individual features
        time_since_last = self._calc_time_since_last(timestamp, history)
        unusual_hour = self._calc_unusual_hour(timestamp)
        session_duration = self._calc_session_duration(user_id, timestamp)
        request_rate = self._calc_request_rate(timestamp, history)
        velocity_change = self._calc_velocity_change(user_id, request_rate)
        tool_changed = self._calc_tool_changed(tool_name, history)
        sequence_entropy = self._calc_sequence_entropy(user_id, tool_name)
        transition_prob = self._calc_transition_prob(tool_name, history)
        sensitivity_score = self._calc_file_sensitivity(params)
        depth_score = self._calc_resource_depth(params)
        new_ip = self._calc_new_ip(user_id, ip)
        size_anomaly = self._calc_request_size_anomaly(params)
        
        # Update state
        self._update_state(user_id, tool_name, request_rate, ip)
        
        return FeatureVector(
            time_since_last=time_since_last,
            unusual_hour=unusual_hour,
            session_duration=session_duration,
            request_rate_per_min=request_rate,
            velocity_change=velocity_change,
            tool_changed=tool_changed,
            sequence_entropy=sequence_entropy,
            tool_transition_prob=transition_prob,
            file_sensitivity_score=sensitivity_score,
            resource_depth_score=depth_score,
            new_ip=new_ip,
            request_size_anomaly=size_anomaly,
        )
    
    def _init_user_state(self, user_id: str, timestamp: datetime) -> None:
        """Initialize state for new user."""
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = timestamp
            self._user_ips[user_id] = set()
            self._user_tool_history[user_id] = deque(maxlen=self.window_size)
            self._user_rate_history[user_id] = deque(maxlen=10)
    
    def _calc_time_since_last(
        self,
        timestamp: datetime,
        history: List[Dict[str, Any]],
    ) -> float:
        """Calculate seconds since last request."""
        if not history:
            return 3600.0  # Default 1 hour if no history
        
        last_ts = history[-1].get("timestamp")
        if isinstance(last_ts, str):
            last_ts = datetime.fromisoformat(last_ts)
        
        if last_ts:
            delta = (timestamp - last_ts).total_seconds()
            return min(max(delta, 0), 7200)  # Cap at 2 hours
        return 3600.0
    
    def _calc_unusual_hour(self, timestamp: datetime) -> int:
        """Check if request is at unusual hour (night/weekend)."""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Night: before 6 AM or after 10 PM
        is_night = hour < 6 or hour > 22
        # Weekend
        is_weekend = weekday >= 5
        
        return 1 if (is_night or is_weekend) else 0
    
    def _calc_session_duration(self, user_id: str, timestamp: datetime) -> float:
        """Calculate session duration in seconds."""
        session_start = self._user_sessions.get(user_id, timestamp)
        duration = (timestamp - session_start).total_seconds()
        
        # Reset session after 4 hours of inactivity (handled elsewhere)
        return min(duration, 14400)  # Cap at 4 hours
    
    def _calc_request_rate(
        self,
        timestamp: datetime,
        history: List[Dict[str, Any]],
    ) -> float:
        """Calculate requests per minute in last minute."""
        one_min_ago = timestamp - timedelta(minutes=1)
        
        count = 0
        for req in history:
            req_ts = req.get("timestamp")
            if isinstance(req_ts, str):
                req_ts = datetime.fromisoformat(req_ts)
            if req_ts and req_ts > one_min_ago:
                count += 1
        
        return min(count, 60)  # Cap at 60/min
    
    def _calc_velocity_change(self, user_id: str, current_rate: float) -> float:
        """Calculate rate of change in request rate (acceleration)."""
        rate_history = self._user_rate_history.get(user_id, deque())
        
        if len(rate_history) < 2:
            return 0.0
        
        # Calculate acceleration (second derivative)
        prev_rate = rate_history[-1]
        velocity = current_rate - prev_rate
        
        if len(rate_history) >= 2:
            prev_velocity = rate_history[-1] - rate_history[-2]
            acceleration = velocity - prev_velocity
            return max(-10, min(10, acceleration))  # Cap
        
        return max(-10, min(10, velocity))
    
    def _calc_tool_changed(
        self,
        tool_name: str,
        history: List[Dict[str, Any]],
    ) -> int:
        """Check if tool changed from last request."""
        if not history:
            return 0
        
        last_tool = history[-1].get("tool") or history[-1].get("tool_name")
        return 1 if last_tool != tool_name else 0
    
    def _calc_sequence_entropy(self, user_id: str, tool_name: str) -> float:
        """Calculate Shannon entropy of tool sequence."""
        tool_history = self._user_tool_history.get(user_id, deque())
        
        if len(tool_history) < 5:
            return 0.5  # Default moderate entropy
        
        # Include current tool
        tools = list(tool_history) + [tool_name]
        
        # Calculate frequency
        counts = Counter(tools)
        total = len(tools)
        
        # Shannon entropy
        entropy = 0.0
        for count in counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # Normalize to 0-1 (max entropy = log2(n_unique))
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        
        return min(1.0, normalized)
    
    def _calc_transition_prob(
        self,
        tool_name: str,
        history: List[Dict[str, Any]],
    ) -> float:
        """Calculate Markov transition probability."""
        if not history:
            return 0.5  # Default
        
        last_tool = history[-1].get("tool") or history[-1].get("tool_name")
        if not last_tool:
            return 0.5
        
        # Look up transition probability
        key = (last_tool, tool_name)
        prob = self.transitions.get(key, 0.1)  # Low default for unseen transitions
        
        return prob
    
    def _calc_file_sensitivity(self, params: Dict[str, Any]) -> float:
        """Calculate file sensitivity score from path patterns."""
        path = params.get("path") or params.get("file_path") or ""
        
        if not path:
            # Check for patterns in any string parameter
            for value in params.values():
                if isinstance(value, str) and "/" in value:
                    path = value
                    break
        
        if not path:
            return 0.0
        
        # Check patterns in order of severity
        for pattern in COMPILED_PATTERNS["critical"]:
            if pattern.search(path):
                return 1.0
        
        for pattern in COMPILED_PATTERNS["high"]:
            if pattern.search(path):
                return 0.7
        
        for pattern in COMPILED_PATTERNS["medium"]:
            if pattern.search(path):
                return 0.4
        
        return 0.1  # Default low sensitivity
    
    def _calc_resource_depth(self, params: Dict[str, Any]) -> float:
        """Calculate directory depth of accessed resource."""
        path = params.get("path") or params.get("file_path") or ""
        
        if not path:
            return 0.0
        
        # Count path segments
        segments = path.replace("\\", "/").split("/")
        segments = [s for s in segments if s]  # Remove empty
        depth = len(segments)
        
        # Normalize (0-10 depth -> 0-1)
        return min(depth / 10.0, 1.0)
    
    def _calc_new_ip(self, user_id: str, ip: Optional[str]) -> int:
        """Check if IP is new for this user."""
        if not ip:
            return 0
        
        known_ips = self._user_ips.get(user_id, set())
        return 1 if ip not in known_ips else 0
    
    def _calc_request_size_anomaly(self, params: Dict[str, Any]) -> float:
        """Calculate Z-score of request size."""
        # Estimate request size from params
        try:
            size = len(str(params))
        except Exception:
            size = 100  # Default
        
        # Add to history
        self._request_sizes.append(size)
        
        if len(self._request_sizes) < 10:
            return 0.0
        
        # Calculate Z-score
        sizes = list(self._request_sizes)
        mean = np.mean(sizes)
        std = np.std(sizes)
        
        if std == 0:
            return 0.0
        
        z_score = (size - mean) / std
        return max(-3, min(3, z_score))  # Cap at ±3
    
    def _update_state(
        self,
        user_id: str,
        tool_name: str,
        request_rate: float,
        ip: Optional[str],
    ) -> None:
        """Update internal state after feature extraction."""
        # Update tool history
        if user_id in self._user_tool_history:
            self._user_tool_history[user_id].append(tool_name)
        
        # Update rate history
        if user_id in self._user_rate_history:
            self._user_rate_history[user_id].append(request_rate)
        
        # Update known IPs
        if ip and user_id in self._user_ips:
            self._user_ips[user_id].add(ip)
    
    def update_transitions(self, transition_counts: Dict[Tuple[str, str], int]) -> None:
        """Update transition probabilities from observed data."""
        # Group by source tool
        source_totals: Dict[str, int] = {}
        for (src, dst), count in transition_counts.items():
            source_totals[src] = source_totals.get(src, 0) + count
        
        # Calculate probabilities
        for (src, dst), count in transition_counts.items():
            total = source_totals[src]
            if total > 0:
                self.transitions[(src, dst)] = count / total
        
        logger.info("transitions_updated", count=len(self.transitions))
    
    def reset_session(self, user_id: str) -> None:
        """Reset session for user (e.g., after long inactivity)."""
        if user_id in self._user_sessions:
            del self._user_sessions[user_id]
        if user_id in self._user_tool_history:
            self._user_tool_history[user_id].clear()
        if user_id in self._user_rate_history:
            self._user_rate_history[user_id].clear()


# Singleton instance
feature_extractor = BehavioralFeatureExtractor()
