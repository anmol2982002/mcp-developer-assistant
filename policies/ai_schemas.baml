# BAML AI Output Schemas
# Defines schemas for AI/LLM output validation

# Code Review Issue
class ReviewIssue {
  file: string @description("File path where issue was found")
  severity: enum<HIGH, MEDIUM, LOW> @description("Issue severity level")
  message: string @description("Clear description of the issue")
  line: int? @description("Line number if applicable")
  suggestion: string? @description("Suggested fix")
}

# Code Review Output
class CodeReviewOutput {
  summary: string @description("One-line summary of changes")
  issues: list<ReviewIssue> @description("List of identified issues")
  test_suggestions: list<string> @description("Suggested test cases")
  risk_level: enum<HIGH, MEDIUM, LOW> @description("Overall risk assessment")
  estimated_review_time_minutes: int @description("Estimated review time")

  # Validation constraints
  @assert(this.issues.length <= 50, "Too many issues reported")
  @assert(this.estimated_review_time_minutes > 0, "Time must be positive")
  @assert(this.estimated_review_time_minutes <= 240, "Max 4 hours")
}

# Intent Check Result
class IntentCheckResult {
  is_valid: bool @description("Whether request matches intended use")
  confidence: float @description("Confidence score 0.0-1.0")
  reason: string @description("Explanation of decision")

  @assert(this.confidence >= 0.0 && this.confidence <= 1.0)
}

# Anomaly Detection Result
class AnomalyResult {
  is_anomalous: bool @description("Whether behavior is anomalous")
  score: float @description("Anomaly score 0.0-1.0")
  reason: string @description("Human-readable explanation")
  features: map<string, float>? @description("Contributing features")

  @assert(this.score >= 0.0 && this.score <= 1.0)
}

# Code Summary Output
class CodeSummaryOutput {
  purpose: string @description("What the code does")
  key_components: list<string> @description("Main modules/classes")
  technologies: list<string> @description("Languages/frameworks used")
  entry_point: string? @description("Main entry point if applicable")
}

# Validate review output and check for secrets
function validate_review_output(json_str: string) -> CodeReviewOutput? {
  output = parse_json(json_str) as CodeReviewOutput

  # Ensure no sensitive data in output
  let sensitive_patterns = [
    "API_KEY",
    "SECRET",
    "PASSWORD",
    "TOKEN",
    "CREDENTIAL"
  ]

  for pattern in sensitive_patterns {
    if contains(output.summary, pattern) {
      return null  # Reject if secrets detected
    }
    for suggestion in output.test_suggestions {
      if contains(suggestion, pattern) {
        return null
      }
    }
  }

  return output
}
