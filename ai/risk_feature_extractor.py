"""
Risk Feature Extractor

Comprehensive feature extraction for ML-based code review risk scoring.
Extracts 10+ features including:
- File sensitivity scores
- Change complexity metrics
- Author familiarity with codebase
- Temporal features (time since last review)
- Security pattern detection
"""

import math
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from observability.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# File Sensitivity Patterns
# =============================================================================

SENSITIVITY_PATTERNS = {
    "critical": [
        r"\.env$",
        r"\.env\.",
        r"secrets?\.ya?ml$",
        r"secrets?\.json$",
        r"credentials",
        r"\.pem$",
        r"\.key$",
        r"private_?key",
    ],
    "high": [
        r"config\.py$",
        r"settings\.py$",
        r"auth",
        r"password",
        r"token",
        r"oauth",
        r"jwt",
        r"security",
        r"permission",
        r"access",
        r"\.sql$",
        r"migration",
    ],
    "medium": [
        r"api",
        r"route",
        r"handler",
        r"controller",
        r"middleware",
        r"database",
        r"model",
        r"schema",
    ],
    "low": [
        r"test",
        r"spec",
        r"\.md$",
        r"\.txt$",
        r"\.rst$",
        r"readme",
        r"docs?/",
    ],
}

# Compile patterns for efficiency
COMPILED_SENSITIVITY = {
    level: [re.compile(p, re.I) for p in patterns]
    for level, patterns in SENSITIVITY_PATTERNS.items()
}

# Security-related code patterns (higher risk)
SECURITY_CODE_PATTERNS = [
    (r"eval\s*\(", "eval_usage"),
    (r"exec\s*\(", "exec_usage"),
    (r"subprocess\.(call|Popen|run)", "subprocess_usage"),
    (r"os\.system", "os_system_usage"),
    (r"pickle\.(load|loads)", "pickle_deserialize"),
    (r"(api_key|secret_key|password)\s*=\s*['\"]", "hardcoded_credential"),
    (r"cursor\.execute.*%s", "sql_string_format"),
    (r"f['\"].*\{.*\}.*SELECT|INSERT|UPDATE|DELETE", "sql_injection_risk"),
    (r"shell\s*=\s*True", "shell_injection_risk"),
    (r"verify\s*=\s*False", "ssl_verify_disabled"),
    (r"disable.*csrf", "csrf_disabled"),
    (r"allow.*origin.*\*", "cors_wildcard"),
]

COMPILED_SECURITY_PATTERNS = [(re.compile(p, re.I), name) for p, name in SECURITY_CODE_PATTERNS]


@dataclass
class RiskFeatures:
    """
    Complete feature vector for ML-based risk prediction.
    
    Contains 12+ features across categories:
    - File sensitivity
    - Change complexity
    - Author familiarity
    - Temporal features
    - Security patterns
    """
    
    # File sensitivity features (0-1 scale)
    file_sensitivity_score: float = 0.0
    sensitive_file_count: int = 0
    critical_file_touched: bool = False
    
    # Change complexity features
    lines_added: int = 0
    lines_removed: int = 0
    files_changed: int = 0
    complexity_score: float = 0.0  # Derived from change size
    
    # Author familiarity features (0-1 scale)
    author_familiarity: float = 0.5  # Default to neutral
    author_file_ownership: float = 0.0  # % of changed files author has touched
    author_recent_activity: float = 0.0  # Activity in last 30 days
    
    # Temporal features
    days_since_last_review: int = 0
    is_weekend_commit: bool = False
    is_late_night_commit: bool = False
    time_risk_score: float = 0.0
    
    # Security pattern features
    security_pattern_count: int = 0
    security_patterns_found: List[str] = field(default_factory=list)
    new_dependency_count: int = 0
    
    # Diff structure features
    has_binary_files: bool = False
    has_large_additions: bool = False  # >200 lines in single file
    
    def to_array(self) -> List[float]:
        """Convert to array for ML model input."""
        return [
            self.file_sensitivity_score,
            float(self.sensitive_file_count),
            float(self.critical_file_touched),
            float(self.lines_added),
            float(self.lines_removed),
            float(self.files_changed),
            self.complexity_score,
            self.author_familiarity,
            self.author_file_ownership,
            self.author_recent_activity,
            float(self.days_since_last_review),
            float(self.is_weekend_commit),
            float(self.is_late_night_commit),
            self.time_risk_score,
            float(self.security_pattern_count),
            float(self.new_dependency_count),
            float(self.has_binary_files),
            float(self.has_large_additions),
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_sensitivity_score": self.file_sensitivity_score,
            "sensitive_file_count": self.sensitive_file_count,
            "critical_file_touched": self.critical_file_touched,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "files_changed": self.files_changed,
            "complexity_score": self.complexity_score,
            "author_familiarity": self.author_familiarity,
            "author_file_ownership": self.author_file_ownership,
            "author_recent_activity": self.author_recent_activity,
            "days_since_last_review": self.days_since_last_review,
            "is_weekend_commit": self.is_weekend_commit,
            "is_late_night_commit": self.is_late_night_commit,
            "time_risk_score": self.time_risk_score,
            "security_pattern_count": self.security_pattern_count,
            "security_patterns_found": self.security_patterns_found,
            "new_dependency_count": self.new_dependency_count,
            "has_binary_files": self.has_binary_files,
            "has_large_additions": self.has_large_additions,
        }
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered feature names for ML model."""
        return [
            "file_sensitivity_score",
            "sensitive_file_count",
            "critical_file_touched",
            "lines_added",
            "lines_removed",
            "files_changed",
            "complexity_score",
            "author_familiarity",
            "author_file_ownership",
            "author_recent_activity",
            "days_since_last_review",
            "is_weekend_commit",
            "is_late_night_commit",
            "time_risk_score",
            "security_pattern_count",
            "new_dependency_count",
            "has_binary_files",
            "has_large_additions",
        ]


class RiskFeatureExtractor:
    """
    Extract comprehensive risk features from code changes.
    
    Used by the risk prediction ML model to assess code review priority
    and routing decisions.
    """
    
    def __init__(self, git_history_days: int = 90):
        """
        Initialize extractor.
        
        Args:
            git_history_days: Days of git history to analyze for familiarity
        """
        self.git_history_days = git_history_days
        self._author_cache: Dict[str, Dict[str, Any]] = {}
    
    def extract_from_diff(
        self,
        diff: str,
        repo_path: str = ".",
        author: Optional[str] = None,
        commit_time: Optional[datetime] = None,
    ) -> RiskFeatures:
        """
        Extract all risk features from a git diff.
        
        Args:
            diff: Git diff content
            repo_path: Path to repository (for git history analysis)
            author: Commit author email/name
            commit_time: Commit timestamp (defaults to now)
            
        Returns:
            RiskFeatures with all extracted features
        """
        features = RiskFeatures()
        commit_time = commit_time or datetime.now()
        
        # Parse diff structure
        changed_files = self._parse_changed_files(diff)
        features.files_changed = len(changed_files)
        
        # Extract line changes
        features.lines_added, features.lines_removed = self._count_line_changes(diff)
        
        # Calculate complexity
        features.complexity_score = self._calc_complexity_score(
            features.lines_added,
            features.lines_removed,
            features.files_changed,
        )
        features.has_large_additions = self._has_large_additions(diff)
        features.has_binary_files = "Binary files" in diff
        
        # File sensitivity analysis
        sensitivity_result = self._calc_file_sensitivity(changed_files)
        features.file_sensitivity_score = sensitivity_result["score"]
        features.sensitive_file_count = sensitivity_result["count"]
        features.critical_file_touched = sensitivity_result["critical"]
        
        # Security pattern detection
        security_result = self._detect_security_patterns(diff)
        features.security_pattern_count = security_result["count"]
        features.security_patterns_found = security_result["patterns"]
        
        # Dependency changes
        features.new_dependency_count = self._count_new_dependencies(diff, changed_files)
        
        # Temporal features
        temporal = self._calc_temporal_features(commit_time)
        features.is_weekend_commit = temporal["is_weekend"]
        features.is_late_night_commit = temporal["is_late_night"]
        features.time_risk_score = temporal["risk_score"]
        
        # Author familiarity (requires git access)
        if author and repo_path:
            try:
                familiarity = self._calc_author_familiarity(author, changed_files, repo_path)
                features.author_familiarity = familiarity["familiarity"]
                features.author_file_ownership = familiarity["ownership"]
                features.author_recent_activity = familiarity["recent_activity"]
            except Exception as e:
                logger.warning("author_familiarity_failed", error=str(e))
        
        # Days since last review (approximated by last commit to these files)
        try:
            features.days_since_last_review = self._calc_days_since_last_change(
                changed_files, repo_path
            )
        except Exception as e:
            logger.warning("days_since_review_failed", error=str(e))
        
        logger.info(
            "risk_features_extracted",
            files=features.files_changed,
            sensitivity=round(features.file_sensitivity_score, 2),
            security_patterns=features.security_pattern_count,
        )
        
        return features
    
    def _parse_changed_files(self, diff: str) -> List[str]:
        """Extract list of changed files from diff."""
        files = []
        
        # Match "diff --git a/path b/path" or "+++ b/path"
        patterns = [
            re.compile(r"^diff --git a/.+ b/(.+)$", re.M),
            re.compile(r"^\+\+\+ b/(.+)$", re.M),
        ]
        
        for pattern in patterns:
            files.extend(pattern.findall(diff))
        
        return list(set(files))
    
    def _count_line_changes(self, diff: str) -> Tuple[int, int]:
        """Count added and removed lines."""
        additions = 0
        deletions = 0
        
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        
        return additions, deletions
    
    def _calc_complexity_score(
        self,
        additions: int,
        deletions: int,
        files: int,
    ) -> float:
        """
        Calculate complexity score (0-1) based on change size.
        
        Uses logarithmic scaling to handle large changes.
        """
        total_lines = additions + deletions
        
        # Log-scale the line count (makes scoring more gradual)
        if total_lines == 0:
            line_score = 0.0
        else:
            line_score = min(1.0, math.log10(total_lines + 1) / 4)  # 10000 lines = 1.0
        
        # File count factor
        file_score = min(1.0, files / 20)  # 20 files = 1.0
        
        # Combine with weights
        return 0.7 * line_score + 0.3 * file_score
    
    def _has_large_additions(self, diff: str, threshold: int = 200) -> bool:
        """Check if any single file has large additions."""
        current_file_additions = 0
        
        for line in diff.splitlines():
            if line.startswith("diff --git"):
                if current_file_additions > threshold:
                    return True
                current_file_additions = 0
            elif line.startswith("+") and not line.startswith("+++"):
                current_file_additions += 1
        
        return current_file_additions > threshold
    
    def _calc_file_sensitivity(self, files: List[str]) -> Dict[str, Any]:
        """
        Calculate file sensitivity score based on patterns.
        
        Returns dict with:
        - score: 0-1 sensitivity score
        - count: number of sensitive files
        - critical: whether any critical files touched
        """
        if not files:
            return {"score": 0.0, "count": 0, "critical": False}
        
        sensitivity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}
        total_score = 0.0
        sensitive_count = 0
        critical_touched = False
        
        for filepath in files:
            file_sensitivity = 0.0
            
            for level, patterns in COMPILED_SENSITIVITY.items():
                if any(p.search(filepath) for p in patterns):
                    file_sensitivity = max(file_sensitivity, sensitivity_weights[level])
                    if level == "critical":
                        critical_touched = True
                    if level in ("critical", "high"):
                        sensitive_count += 1
                    break
            
            total_score += file_sensitivity
        
        avg_score = total_score / len(files) if files else 0.0
        
        return {
            "score": min(1.0, avg_score),
            "count": sensitive_count,
            "critical": critical_touched,
        }
    
    def _detect_security_patterns(self, diff: str) -> Dict[str, Any]:
        """
        Detect security-related patterns in added lines.
        
        Only scans added lines (not removed) for new security concerns.
        """
        # Extract only added lines
        added_lines = "\n".join(
            line[1:] for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        
        patterns_found = []
        for pattern, name in COMPILED_SECURITY_PATTERNS:
            if pattern.search(added_lines):
                patterns_found.append(name)
        
        return {
            "count": len(patterns_found),
            "patterns": patterns_found,
        }
    
    def _count_new_dependencies(self, diff: str, files: List[str]) -> int:
        """Count newly added dependencies in common dependency files."""
        dep_files = ["requirements.txt", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
        count = 0
        
        for dep_file in dep_files:
            if any(dep_file in f for f in files):
                # Count added lines in dependency files
                in_dep_section = False
                for line in diff.splitlines():
                    if dep_file in line:
                        in_dep_section = True
                    elif line.startswith("diff --git"):
                        in_dep_section = False
                    elif in_dep_section and line.startswith("+") and not line.startswith("+++"):
                        # Simple heuristic: new lines in dep files are new deps
                        if "=" in line or ":" in line or "@" in line:
                            count += 1
        
        return count
    
    def _calc_temporal_features(self, commit_time: datetime) -> Dict[str, Any]:
        """Calculate temporal risk features."""
        is_weekend = commit_time.weekday() >= 5
        hour = commit_time.hour
        is_late_night = hour < 6 or hour >= 22
        
        # Risk score based on time
        time_risk = 0.0
        if is_weekend:
            time_risk += 0.3
        if is_late_night:
            time_risk += 0.4
        elif hour < 9 or hour >= 18:  # Outside business hours
            time_risk += 0.1
        
        return {
            "is_weekend": is_weekend,
            "is_late_night": is_late_night,
            "risk_score": min(1.0, time_risk),
        }
    
    def _calc_author_familiarity(
        self,
        author: str,
        changed_files: List[str],
        repo_path: str,
    ) -> Dict[str, float]:
        """
        Calculate author familiarity with changed files.
        
        Uses git history to determine:
        - Overall author contribution to repo
        - Author ownership of changed files
        - Recent activity level
        """
        # Use cache if available
        cache_key = f"{author}:{repo_path}"
        if cache_key in self._author_cache:
            cached = self._author_cache[cache_key]
            # Recalculate ownership for these specific files
            ownership = self._calc_file_ownership(author, changed_files, repo_path)
            cached["ownership"] = ownership
            return cached
        
        result = {
            "familiarity": 0.5,
            "ownership": 0.0,
            "recent_activity": 0.0,
        }
        
        try:
            # Get author's commit count vs total
            since_date = (datetime.now() - timedelta(days=self.git_history_days)).isoformat()
            
            # Total commits by author
            author_commits = self._run_git(
                ["log", f"--since={since_date}", f"--author={author}", "--oneline"],
                repo_path,
            )
            author_count = len(author_commits.strip().splitlines()) if author_commits else 0
            
            # Total commits
            all_commits = self._run_git(
                ["log", f"--since={since_date}", "--oneline"],
                repo_path,
            )
            total_count = len(all_commits.strip().splitlines()) if all_commits else 1
            
            # Familiarity = author's share of commits
            result["familiarity"] = min(1.0, (author_count / total_count) * 2)  # Scale up
            
            # Recent activity (last 7 days)
            recent_date = (datetime.now() - timedelta(days=7)).isoformat()
            recent_commits = self._run_git(
                ["log", f"--since={recent_date}", f"--author={author}", "--oneline"],
                repo_path,
            )
            recent_count = len(recent_commits.strip().splitlines()) if recent_commits else 0
            result["recent_activity"] = min(1.0, recent_count / 10)  # 10 commits = max
            
            # File ownership
            result["ownership"] = self._calc_file_ownership(author, changed_files, repo_path)
            
            # Cache result
            self._author_cache[cache_key] = result
            
        except Exception as e:
            logger.warning("git_history_failed", error=str(e))
        
        return result
    
    def _calc_file_ownership(
        self,
        author: str,
        files: List[str],
        repo_path: str,
    ) -> float:
        """Calculate what % of files the author has previously modified."""
        if not files:
            return 0.0
        
        owned_count = 0
        for filepath in files:
            try:
                # Check if author has touched this file before
                result = self._run_git(
                    ["log", "--oneline", f"--author={author}", "--", filepath],
                    repo_path,
                )
                if result and result.strip():
                    owned_count += 1
            except Exception:
                pass
        
        return owned_count / len(files)
    
    def _calc_days_since_last_change(self, files: List[str], repo_path: str) -> int:
        """Calculate days since any of these files were last changed."""
        if not files:
            return 0
        
        oldest_days = 0
        for filepath in files:
            try:
                result = self._run_git(
                    ["log", "-1", "--format=%ci", "--", filepath],
                    repo_path,
                )
                if result and result.strip():
                    date_str = result.strip().split()[0]  # Get date part
                    last_date = datetime.strptime(date_str, "%Y-%m-%d")
                    days = (datetime.now() - last_date).days
                    oldest_days = max(oldest_days, days)
            except Exception:
                pass
        
        return oldest_days
    
    def _run_git(self, args: List[str], cwd: str) -> Optional[str]:
        """Run git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None


# Singleton instance
risk_feature_extractor = RiskFeatureExtractor()
