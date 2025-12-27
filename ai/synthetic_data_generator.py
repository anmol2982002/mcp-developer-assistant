"""
Synthetic Data Generator

Generates realistic training data for anomaly detection cold start.
Creates normal behavior patterns and injects anomalies for balanced training.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai.feature_extractor import FeatureVector
from observability.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UserProfile:
    """User behavior profile for synthetic data generation."""
    
    name: str
    work_hours: Tuple[int, int]  # (start_hour, end_hour)
    avg_requests_per_hour: float
    common_tools: List[str]
    tool_weights: List[float]
    request_burst_prob: float  # Probability of burst requests
    sensitivity_access_prob: float  # Probability of accessing sensitive files


# Predefined user profiles representing different developer types
USER_PROFILES = {
    "regular_developer": UserProfile(
        name="regular_developer",
        work_hours=(9, 18),
        avg_requests_per_hour=15.0,
        common_tools=["read_file", "search_files", "git_status", "git_diff", "list_dir"],
        tool_weights=[0.35, 0.25, 0.15, 0.15, 0.10],
        request_burst_prob=0.05,
        sensitivity_access_prob=0.02,
    ),
    "senior_engineer": UserProfile(
        name="senior_engineer",
        work_hours=(8, 20),
        avg_requests_per_hour=25.0,
        common_tools=["read_file", "git_diff", "search_files", "review_changes", "list_dir"],
        tool_weights=[0.30, 0.25, 0.20, 0.15, 0.10],
        request_burst_prob=0.08,
        sensitivity_access_prob=0.05,  # Slightly higher - config review
    ),
    "ops_engineer": UserProfile(
        name="ops_engineer",
        work_hours=(6, 22),  # Wider hours for on-call
        avg_requests_per_hour=10.0,
        common_tools=["read_file", "list_dir", "git_status", "search_files"],
        tool_weights=[0.40, 0.30, 0.20, 0.10],
        request_burst_prob=0.10,  # More likely during incidents
        sensitivity_access_prob=0.08,  # Config access
    ),
    "new_developer": UserProfile(
        name="new_developer",
        work_hours=(9, 17),
        avg_requests_per_hour=30.0,  # Higher - exploring codebase
        common_tools=["read_file", "list_dir", "search_files", "git_log"],
        tool_weights=[0.45, 0.25, 0.20, 0.10],
        request_burst_prob=0.03,
        sensitivity_access_prob=0.01,
    ),
}

# Common file paths for synthetic data
COMMON_FILE_PATHS = [
    "src/main.py",
    "src/utils.py",
    "src/models.py",
    "src/api/routes.py",
    "src/services/auth.py",
    "src/services/user.py",
    "tests/test_main.py",
    "tests/test_utils.py",
    "README.md",
    "requirements.txt",
    "docker-compose.yml",
    "Dockerfile",
    "config/settings.py",
    "scripts/deploy.sh",
]

SENSITIVE_FILE_PATHS = [
    ".env",
    ".env.production",
    "secrets/api_key.txt",
    "config/credentials.yaml",
    ".git/config",
    "id_rsa",
]


class SyntheticDataGenerator:
    """
    Generate synthetic training data for anomaly detection.
    
    Creates realistic user behavior patterns with configurable anomaly injection.
    """
    
    def __init__(
        self,
        profiles: Optional[Dict[str, UserProfile]] = None,
        random_seed: int = 42,
    ):
        """
        Initialize generator.
        
        Args:
            profiles: User profiles to use (defaults to built-in profiles)
            random_seed: Random seed for reproducibility
        """
        self.profiles = profiles or USER_PROFILES
        self.rng = np.random.default_rng(random_seed)
        random.seed(random_seed)
    
    def generate_normal_session(
        self,
        user_id: str,
        profile_name: str = "regular_developer",
        session_duration_hours: float = 4.0,
        start_time: Optional[datetime] = None,
    ) -> List[Dict[str, float]]:
        """
        Generate normal behavior session for a user.
        
        Args:
            user_id: User identifier
            profile_name: Profile to use for behavior
            session_duration_hours: Session length
            start_time: Session start time
            
        Returns:
            List of feature dictionaries
        """
        profile = self.profiles.get(profile_name, USER_PROFILES["regular_developer"])
        
        if start_time is None:
            # Random day, within work hours
            base_date = datetime.now().replace(hour=profile.work_hours[0], minute=0, second=0)
            start_time = base_date + timedelta(
                days=-self.rng.integers(0, 30),
                hours=self.rng.uniform(0, profile.work_hours[1] - profile.work_hours[0] - session_duration_hours),
            )
        
        features_list = []
        current_time = start_time
        end_time = start_time + timedelta(hours=session_duration_hours)
        
        last_tool = None
        request_times = []
        
        while current_time < end_time:
            # Inter-request time (exponential distribution)
            avg_interval = 3600.0 / profile.avg_requests_per_hour
            interval = self.rng.exponential(avg_interval)
            
            # Occasional bursts
            if self.rng.random() < profile.request_burst_prob:
                interval = self.rng.uniform(0.5, 3.0)
            
            current_time += timedelta(seconds=interval)
            if current_time >= end_time:
                break
            
            # Select tool based on weights
            tool = self.rng.choice(profile.common_tools, p=profile.tool_weights)
            
            # Generate file path
            if self.rng.random() < profile.sensitivity_access_prob:
                file_path = self.rng.choice(SENSITIVE_FILE_PATHS)
                sensitivity = 0.8 + self.rng.uniform(0, 0.2)
            else:
                file_path = self.rng.choice(COMMON_FILE_PATHS)
                sensitivity = 0.1 + self.rng.uniform(0, 0.2)
            
            # Calculate features
            time_since_last = interval
            tool_changed = 1 if last_tool and last_tool != tool else 0
            
            # Request rate (requests in last minute)
            one_min_ago = current_time - timedelta(minutes=1)
            rate = len([t for t in request_times if t > one_min_ago])
            
            # Unusual hour
            hour = current_time.hour
            unusual_hour = 1 if hour < 6 or hour > 22 else 0
            
            # Session duration
            session_duration = (current_time - start_time).total_seconds()
            
            # Velocity (rate change) - simplified
            if len(features_list) >= 2:
                prev_rate = features_list[-1].get("request_rate_per_min", 0)
                velocity = rate - prev_rate
            else:
                velocity = 0.0
            
            # Sequence entropy - simplified (random for normal)
            entropy = 0.3 + self.rng.uniform(0, 0.3)
            
            # Transition probability - higher for normal sequences
            transition_prob = 0.3 + self.rng.uniform(0, 0.4)
            
            # Depth score
            depth = len(file_path.split("/")) / 10.0
            
            features = {
                "time_since_last": min(time_since_last, 7200),
                "unusual_hour": float(unusual_hour),
                "session_duration": min(session_duration, 14400),
                "request_rate_per_min": min(rate, 60),
                "velocity_change": max(-10, min(10, velocity)),
                "tool_changed": float(tool_changed),
                "sequence_entropy": entropy,
                "tool_transition_prob": transition_prob,
                "file_sensitivity_score": sensitivity,
                "resource_depth_score": min(depth, 1.0),
                "new_ip": 0.0,  # No new IPs for normal sessions
                "request_size_anomaly": self.rng.normal(0, 0.5),
            }
            
            features_list.append(features)
            request_times.append(current_time)
            last_tool = tool
        
        return features_list
    
    def generate_anomalous_session(
        self,
        user_id: str,
        anomaly_type: str = "random",
        base_profile: str = "regular_developer",
        session_duration_hours: float = 1.0,
    ) -> List[Dict[str, float]]:
        """
        Generate anomalous behavior session.
        
        Args:
            user_id: User identifier
            anomaly_type: Type of anomaly (rapid_fire, night_access, sensitive_access, new_ip)
            base_profile: Base profile to deviate from
            session_duration_hours: Session length
            
        Returns:
            List of anomalous feature dictionaries
        """
        if anomaly_type == "random":
            anomaly_type = self.rng.choice([
                "rapid_fire", "night_access", "sensitive_access", 
                "new_ip", "high_entropy", "unusual_tools"
            ])
        
        features_list = []
        
        if anomaly_type == "rapid_fire":
            # Very rapid requests
            n_requests = self.rng.integers(20, 50)
            for i in range(n_requests):
                features = self._base_features()
                features["time_since_last"] = self.rng.uniform(0.1, 1.0)  # < 1 second
                features["request_rate_per_min"] = self.rng.uniform(15, 40)
                features["velocity_change"] = self.rng.uniform(3, 8)
                features_list.append(features)
        
        elif anomaly_type == "night_access":
            # Access during unusual hours
            n_requests = self.rng.integers(10, 30)
            for i in range(n_requests):
                features = self._base_features()
                features["unusual_hour"] = 1.0
                features["time_since_last"] = self.rng.exponential(60)
                features_list.append(features)
        
        elif anomaly_type == "sensitive_access":
            # Accessing many sensitive files
            n_requests = self.rng.integers(15, 40)
            for i in range(n_requests):
                features = self._base_features()
                features["file_sensitivity_score"] = self.rng.uniform(0.7, 1.0)
                features["resource_depth_score"] = self.rng.uniform(0.5, 0.9)
                features_list.append(features)
        
        elif anomaly_type == "new_ip":
            # Multiple requests from new IP
            n_requests = self.rng.integers(10, 25)
            for i in range(n_requests):
                features = self._base_features()
                features["new_ip"] = 1.0
                features_list.append(features)
        
        elif anomaly_type == "high_entropy":
            # Random tool access patterns
            n_requests = self.rng.integers(20, 40)
            for i in range(n_requests):
                features = self._base_features()
                features["sequence_entropy"] = self.rng.uniform(0.85, 1.0)
                features["tool_transition_prob"] = self.rng.uniform(0.0, 0.15)
                features["tool_changed"] = 1.0  # Always changing tools
                features_list.append(features)
        
        elif anomaly_type == "unusual_tools":
            # Using tools in unusual sequence
            n_requests = self.rng.integers(15, 35)
            for i in range(n_requests):
                features = self._base_features()
                features["tool_transition_prob"] = self.rng.uniform(0.0, 0.1)
                features["sequence_entropy"] = self.rng.uniform(0.7, 0.95)
                features_list.append(features)
        
        return features_list
    
    def _base_features(self) -> Dict[str, float]:
        """Generate base features with slight randomness."""
        return {
            "time_since_last": self.rng.exponential(30),
            "unusual_hour": 0.0,
            "session_duration": self.rng.uniform(300, 3600),
            "request_rate_per_min": self.rng.uniform(1, 5),
            "velocity_change": self.rng.normal(0, 1),
            "tool_changed": float(self.rng.random() < 0.3),
            "sequence_entropy": self.rng.uniform(0.3, 0.6),
            "tool_transition_prob": self.rng.uniform(0.2, 0.5),
            "file_sensitivity_score": self.rng.uniform(0.1, 0.3),
            "resource_depth_score": self.rng.uniform(0.1, 0.4),
            "new_ip": 0.0,
            "request_size_anomaly": self.rng.normal(0, 0.5),
        }
    
    def generate_training_dataset(
        self,
        n_normal_sessions: int = 100,
        n_anomaly_sessions: int = 10,
        session_duration_hours: float = 4.0,
    ) -> Tuple[List[Dict[str, float]], List[int]]:
        """
        Generate complete training dataset with labels.
        
        Args:
            n_normal_sessions: Number of normal sessions
            n_anomaly_sessions: Number of anomalous sessions
            session_duration_hours: Average session length
            
        Returns:
            Tuple of (features_list, labels) where label=0 is normal, label=1 is anomaly
        """
        all_features = []
        all_labels = []
        
        # Generate normal sessions
        profile_names = list(self.profiles.keys())
        for i in range(n_normal_sessions):
            user_id = f"user_{i:04d}"
            profile = self.rng.choice(profile_names)
            duration = self.rng.uniform(
                session_duration_hours * 0.5,
                session_duration_hours * 1.5,
            )
            
            session_features = self.generate_normal_session(
                user_id=user_id,
                profile_name=profile,
                session_duration_hours=duration,
            )
            
            all_features.extend(session_features)
            all_labels.extend([0] * len(session_features))
        
        # Generate anomalous sessions
        anomaly_types = ["rapid_fire", "night_access", "sensitive_access", 
                        "new_ip", "high_entropy", "unusual_tools"]
        
        for i in range(n_anomaly_sessions):
            user_id = f"anomaly_user_{i:03d}"
            anomaly_type = self.rng.choice(anomaly_types)
            
            session_features = self.generate_anomalous_session(
                user_id=user_id,
                anomaly_type=anomaly_type,
            )
            
            all_features.extend(session_features)
            all_labels.extend([1] * len(session_features))
        
        logger.info(
            "synthetic_dataset_generated",
            total_samples=len(all_features),
            normal_samples=sum(1 for l in all_labels if l == 0),
            anomaly_samples=sum(1 for l in all_labels if l == 1),
        )
        
        return all_features, all_labels
    
    def to_numpy(
        self,
        features_list: List[Dict[str, float]],
    ) -> np.ndarray:
        """Convert feature list to numpy array."""
        feature_names = FeatureVector.feature_names()
        return np.array([
            [f.get(name, 0.0) for name in feature_names]
            for f in features_list
        ])


# Singleton
synthetic_generator = SyntheticDataGenerator()
