"""
Review Router

Route code reviews based on risk assessment:
- Priority assignment (critical, high, normal, low)
- Suggested reviewers based on file ownership
- Security/senior review requirements
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from observability.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewRouting:
    """
    Review routing decision.
    
    Contains priority, suggested reviewers, and special requirements.
    """
    
    priority: str  # "critical", "high", "normal", "low"
    suggested_reviewers: List[str] = field(default_factory=list)
    estimated_review_time: int = 15  # Minutes
    requires_security_review: bool = False
    requires_senior_review: bool = False
    reason: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "priority": self.priority,
            "suggested_reviewers": self.suggested_reviewers,
            "estimated_review_time": self.estimated_review_time,
            "requires_security_review": self.requires_security_review,
            "requires_senior_review": self.requires_senior_review,
            "reason": self.reason,
        }


class ReviewRouter:
    """
    Route code reviews based on risk assessment.
    
    Uses risk score, security patterns, and file sensitivity to
    determine review priority and requirements.
    """
    
    # Risk thresholds for priority levels
    PRIORITY_THRESHOLDS = {
        "critical": 0.85,
        "high": 0.65,
        "normal": 0.35,
    }
    
    # Base review times (minutes) by priority
    BASE_REVIEW_TIMES = {
        "critical": 60,
        "high": 45,
        "normal": 20,
        "low": 10,
    }
    
    def __init__(
        self,
        security_reviewers: Optional[List[str]] = None,
        senior_reviewers: Optional[List[str]] = None,
    ):
        """
        Initialize router.
        
        Args:
            security_reviewers: List of security team reviewers
            senior_reviewers: List of senior developers
        """
        self.security_reviewers = security_reviewers or []
        self.senior_reviewers = senior_reviewers or []
    
    def determine_route(
        self,
        risk_score: float,
        risk_factors: List[str],
        security_patterns: Optional[List[str]] = None,
        lines_changed: int = 0,
        files_changed: int = 0,
        file_owners: Optional[Dict[str, List[str]]] = None,
    ) -> ReviewRouting:
        """
        Determine review routing based on risk assessment.
        
        Args:
            risk_score: ML-predicted risk score (0-1)
            risk_factors: Contributing risk factors
            security_patterns: Security patterns found in diff
            lines_changed: Total lines changed
            files_changed: Number of files changed
            file_owners: Dict mapping files to their owners/contributors
            
        Returns:
            ReviewRouting with priority and requirements
        """
        security_patterns = security_patterns or []
        file_owners = file_owners or {}
        
        # Determine priority from risk score
        priority = self._calc_priority(risk_score)
        
        # Check security requirements
        requires_security = (
            len(security_patterns) > 0 or
            any("security" in f.lower() for f in risk_factors) or
            any("credential" in f.lower() for f in risk_factors) or
            any("critical" in f.lower() for f in risk_factors)
        )
        
        # Check if needs senior review
        requires_senior = (
            risk_score >= 0.7 or
            files_changed >= 15 or
            lines_changed >= 500 or
            any("architecture" in f.lower() for f in risk_factors)
        )
        
        # Calculate review time
        review_time = self._estimate_review_time(
            priority, lines_changed, files_changed
        )
        
        # Suggest reviewers
        suggested_reviewers = self._suggest_reviewers(
            file_owners,
            requires_security,
            requires_senior,
        )
        
        # Build reason
        reasons = []
        if risk_score >= 0.8:
            reasons.append(f"High risk score ({risk_score:.2f})")
        if requires_security:
            reasons.append(f"Security patterns: {', '.join(security_patterns[:3])}")
        if files_changed >= 10:
            reasons.append(f"Large scope ({files_changed} files)")
        if not reasons:
            reasons.append("Standard review")
        
        routing = ReviewRouting(
            priority=priority,
            suggested_reviewers=suggested_reviewers,
            estimated_review_time=review_time,
            requires_security_review=requires_security,
            requires_senior_review=requires_senior,
            reason=" | ".join(reasons),
        )
        
        logger.info(
            "review_routed",
            priority=priority,
            requires_security=requires_security,
            requires_senior=requires_senior,
            review_time=review_time,
        )
        
        return routing
    
    def _calc_priority(self, risk_score: float) -> str:
        """Calculate priority level from risk score."""
        if risk_score >= self.PRIORITY_THRESHOLDS["critical"]:
            return "critical"
        elif risk_score >= self.PRIORITY_THRESHOLDS["high"]:
            return "high"
        elif risk_score >= self.PRIORITY_THRESHOLDS["normal"]:
            return "normal"
        else:
            return "low"
    
    def _estimate_review_time(
        self,
        priority: str,
        lines_changed: int,
        files_changed: int,
    ) -> int:
        """Estimate review time in minutes."""
        base_time = self.BASE_REVIEW_TIMES.get(priority, 20)
        
        # Add time for size
        line_factor = min(60, lines_changed // 50)  # +1 min per 50 lines
        file_factor = min(30, files_changed * 2)  # +2 min per file
        
        return base_time + line_factor + file_factor
    
    def _suggest_reviewers(
        self,
        file_owners: Dict[str, List[str]],
        requires_security: bool,
        requires_senior: bool,
    ) -> List[str]:
        """Suggest reviewers based on file ownership and requirements."""
        suggested = []
        
        # Add file owners (most commits to these files)
        if file_owners:
            # Flatten and dedupe owners
            all_owners = []
            for owners in file_owners.values():
                all_owners.extend(owners)
            
            # Count occurrences
            owner_counts = {}
            for owner in all_owners:
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
            
            # Sort by count and take top 2
            sorted_owners = sorted(owner_counts.items(), key=lambda x: -x[1])
            suggested.extend([owner for owner, _ in sorted_owners[:2]])
        
        # Add security reviewer if needed
        if requires_security and self.security_reviewers:
            suggested.append(self.security_reviewers[0])
        
        # Add senior reviewer if needed
        if requires_senior and self.senior_reviewers:
            # Avoid duplicates
            for senior in self.senior_reviewers:
                if senior not in suggested:
                    suggested.append(senior)
                    break
        
        return suggested[:3]  # Max 3 suggested reviewers


# Singleton
review_router = ReviewRouter()
