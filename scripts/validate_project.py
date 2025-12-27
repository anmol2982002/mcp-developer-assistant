#!/usr/bin/env python
"""
MCP Developer Assistant - Comprehensive Project Validation Script

This script validates the entire project implementation across all phases:
- Phase 1: MVP Core (File Tools, Git Tools, Code Analysis)
- Phase 2: Security & Proxy (OAuth, Rate Limiting, Audit)
- Phase 3: ML Security (Intent Checking, Anomaly Detection)
- Phase 4: AI Tools (Embeddings, Semantic Search)
- Phase 5: Code Review (Risk Scoring, Review Routing)
- Phase 6: Production Ready (Docker, Observability)

Usage:
    python scripts/validate_project.py [--verbose] [--phase N]
"""

import sys
import os
import importlib
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    name: str
    passed: bool
    message: str
    phase: int
    category: str
    error: Optional[str] = None


@dataclass
class PhaseReport:
    """Report for a single phase."""
    phase: int
    name: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        return (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0.0


class ProjectValidator:
    """Comprehensive project validation."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.phase_reports: Dict[int, PhaseReport] = {
            0: PhaseReport(0, "Project Structure"),
            1: PhaseReport(1, "MVP Core"),
            2: PhaseReport(2, "Security & Proxy"),
            3: PhaseReport(3, "ML Security"),
            4: PhaseReport(4, "AI Tools"),
            5: PhaseReport(5, "Code Review"),
            6: PhaseReport(6, "Production Ready"),
        }
    
    def log(self, msg: str):
        """Log message if verbose mode."""
        if self.verbose:
            print(f"  [DEBUG] {msg}")
    
    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        report = self.phase_reports[result.phase]
        report.results.append(result)
        report.total_tests += 1
        if result.passed:
            report.passed_tests += 1
        else:
            report.failed_tests += 1
    
    def test_import(self, module_path: str, phase: int, category: str) -> bool:
        """Test if a module can be imported."""
        try:
            importlib.import_module(module_path)
            self.add_result(ValidationResult(
                name=f"Import {module_path}",
                passed=True,
                message=f"Successfully imported {module_path}",
                phase=phase,
                category=category
            ))
            return True
        except Exception as e:
            self.add_result(ValidationResult(
                name=f"Import {module_path}",
                passed=False,
                message=f"Failed to import {module_path}",
                phase=phase,
                category=category,
                error=str(e)
            ))
            return False
    
    def test_class_exists(self, module_path: str, class_name: str, 
                          phase: int, category: str) -> bool:
        """Test if a class exists in a module."""
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                self.add_result(ValidationResult(
                    name=f"Class {class_name}",
                    passed=True,
                    message=f"Found {class_name} in {module_path}",
                    phase=phase,
                    category=category
                ))
                return True
            else:
                self.add_result(ValidationResult(
                    name=f"Class {class_name}",
                    passed=False,
                    message=f"{class_name} not found in {module_path}",
                    phase=phase,
                    category=category
                ))
                return False
        except Exception as e:
            self.add_result(ValidationResult(
                name=f"Class {class_name}",
                passed=False,
                message=f"Error checking {class_name}",
                phase=phase,
                category=category,
                error=str(e)
            ))
            return False
    
    def test_file_exists(self, file_path: str, phase: int, category: str) -> bool:
        """Test if a file exists."""
        full_path = PROJECT_ROOT / file_path
        exists = full_path.exists()
        self.add_result(ValidationResult(
            name=f"File {file_path}",
            passed=exists,
            message=f"{'Found' if exists else 'Missing'}: {file_path}",
            phase=phase,
            category=category
        ))
        return exists
    
    # =========================================================================
    # Phase 0: Project Structure
    # =========================================================================
    
    def validate_phase_0(self):
        """Validate project structure."""
        print("\n📁 Phase 0: Project Structure")
        print("-" * 50)
        
        # Core directories
        dirs = [
            "proxy", "server", "ai", "observability", "models",
            "database", "tests", "scripts", "config", "policies"
        ]
        for d in dirs:
            self.test_file_exists(d, 0, "Directories")
        
        # Config files
        configs = [
            "requirements.txt", "Dockerfile", "docker-compose.yml",
            ".env.example", "pyproject.toml"
        ]
        for c in configs:
            self.test_file_exists(c, 0, "Config Files")
    
    # =========================================================================
    # Phase 1: MVP Core
    # =========================================================================
    
    def validate_phase_1(self):
        """Validate Phase 1 - MVP Core."""
        print("\n🔧 Phase 1: MVP Core")
        print("-" * 50)
        
        # Server module
        self.test_import("server.mcp_server", 1, "MCP Server")
        self.test_class_exists("server.mcp_server", "ToolRegistry", 1, "MCP Server")
        
        # File Tools
        self.test_import("server.tools.file_tools", 1, "File Tools")
        self.test_class_exists("server.tools.file_tools", "ReadFileTool", 1, "File Tools")
        self.test_class_exists("server.tools.file_tools", "SearchFilesTool", 1, "File Tools")
        self.test_class_exists("server.tools.file_tools", "ListDirectoryTool", 1, "File Tools")
        
        # Git Tools
        self.test_import("server.tools.git_tools", 1, "Git Tools")
        self.test_class_exists("server.tools.git_tools", "GitStatusTool", 1, "Git Tools")
        self.test_class_exists("server.tools.git_tools", "GitDiffTool", 1, "Git Tools")
        self.test_class_exists("server.tools.git_tools", "GitLogTool", 1, "Git Tools")
        
        # Code Tools
        self.test_import("server.tools.code_tools", 1, "Code Tools")
        self.test_class_exists("server.tools.code_tools", "ExtractFunctionsTool", 1, "Code Tools")
        
        # Policy Engine
        self.test_import("server.policy_engine", 1, "Policy Engine")
        
        # Logging
        self.test_import("observability.logging_config", 1, "Observability")
        self.test_import("observability.metrics", 1, "Observability")
    
    # =========================================================================
    # Phase 2: Security & Proxy
    # =========================================================================
    
    def validate_phase_2(self):
        """Validate Phase 2 - Security & Proxy."""
        print("\n🔐 Phase 2: Security & Proxy")
        print("-" * 50)
        
        # Auth Gateway
        self.test_import("proxy.auth_gateway", 2, "Auth Gateway")
        
        # OAuth Validator
        self.test_import("proxy.oauth_validator", 2, "OAuth")
        self.test_class_exists("proxy.oauth_validator", "OAuth21Validator", 2, "OAuth")
        
        # Rate Limiter
        self.test_import("proxy.rate_limiter", 2, "Rate Limiting")
        self.test_class_exists("proxy.rate_limiter", "SlidingWindowRateLimiter", 2, "Rate Limiting")
        
        # Consent DB
        self.test_import("proxy.consent_db", 2, "Consent")
        self.test_class_exists("proxy.consent_db", "ConsentDB", 2, "Consent")
        
        # Consent Middleware
        self.test_import("proxy.consent_middleware", 2, "Consent")
        
        # Path Sanitizer
        self.test_import("proxy.path_sanitizer", 2, "Security")
        
        # Refresh Token Store
        self.test_import("proxy.refresh_token_store", 2, "OAuth")
    
    # =========================================================================
    # Phase 3: ML Security
    # =========================================================================
    
    def validate_phase_3(self):
        """Validate Phase 3 - ML Security."""
        print("\n🤖 Phase 3: ML Security")
        print("-" * 50)
        
        # Intent Checker
        self.test_import("proxy.intent_checker", 3, "Intent Checking")
        self.test_class_exists("proxy.intent_checker", "IntentChecker", 3, "Intent Checking")
        
        # Anomaly Detector
        self.test_import("proxy.anomaly_detector", 3, "Anomaly Detection")
        self.test_class_exists("proxy.anomaly_detector", "EnsembleAnomalyDetector", 3, "Anomaly Detection")
        
        # LLM Client
        self.test_import("ai.llm_client", 3, "LLM Integration")
        self.test_class_exists("ai.llm_client", "LLMClient", 3, "LLM Integration")
        
        # Feature Extractor
        self.test_import("ai.feature_extractor", 3, "ML Features")
        self.test_class_exists("ai.feature_extractor", "BehavioralFeatureExtractor", 3, "ML Features")
        
        # Model Trainer
        self.test_import("ai.model_trainer", 3, "ML Training")
        self.test_class_exists("ai.model_trainer", "AnomalyModelTrainer", 3, "ML Training")
        
        # Synthetic Data Generator
        self.test_import("ai.synthetic_data_generator", 3, "ML Training")
        
        # SHAP Explainer
        self.test_import("ai.shap_explainer", 3, "ML Explainability")
        
        # Model Registry
        self.test_import("ai.model_registry", 3, "ML Ops")
    
    # =========================================================================
    # Phase 4: AI Tools
    # =========================================================================
    
    def validate_phase_4(self):
        """Validate Phase 4 - AI Tools."""
        print("\n🧠 Phase 4: AI Tools")
        print("-" * 50)
        
        # Embedding Manager
        self.test_import("ai.embedding_manager", 4, "Embeddings")
        self.test_class_exists("ai.embedding_manager", "EmbeddingManager", 4, "Embeddings")
        
        # Hybrid Search
        self.test_import("ai.hybrid_search", 4, "Semantic Search")
        self.test_class_exists("ai.hybrid_search", "HybridSearchEngine", 4, "Semantic Search")
        
        # File Watcher
        self.test_import("ai.file_watcher", 4, "Indexing")
        
        # AI Tools
        self.test_import("server.tools.ai_tools", 4, "AI Tools")
        self.test_class_exists("server.tools.ai_tools", "AskAboutCodeTool", 4, "AI Tools")
        self.test_class_exists("server.tools.ai_tools", "SummarizeRepoTool", 4, "AI Tools")
        self.test_class_exists("server.tools.ai_tools", "SummarizeDiffTool", 4, "AI Tools")
        
        # Output Validator
        self.test_import("ai.output_validator", 4, "Output Validation")
    
    # =========================================================================
    # Phase 5: Code Review
    # =========================================================================
    
    def validate_phase_5(self):
        """Validate Phase 5 - Code Review."""
        print("\n📝 Phase 5: Code Review")
        print("-" * 50)
        
        # Code Review Tool
        self.test_class_exists("server.tools.ai_tools", "ReviewChangesTool", 5, "Code Review")
        
        # Risk Scorer
        self.test_import("ai.risk_scorer", 5, "Risk Scoring")
        self.test_class_exists("ai.risk_scorer", "RiskScorer", 5, "Risk Scoring")
        
        # Risk Feature Extractor
        self.test_import("ai.risk_feature_extractor", 5, "Risk Features")
        
        # Risk Prediction Model
        self.test_import("ai.risk_prediction_model", 5, "ML Risk Model")
        self.test_class_exists("ai.risk_prediction_model", "RiskPredictionModel", 5, "ML Risk Model")
        
        # Review Router
        self.test_import("ai.review_router", 5, "Review Routing")
    
    # =========================================================================
    # Phase 6: Production Ready
    # =========================================================================
    
    def validate_phase_6(self):
        """Validate Phase 6 - Production Ready."""
        print("\n🚀 Phase 6: Production Ready")
        print("-" * 50)
        
        # Docker
        self.test_file_exists("Dockerfile", 6, "Docker")
        self.test_file_exists("docker-compose.yml", 6, "Docker")
        
        # Health Check
        self.test_import("observability.health_check", 6, "Health")
        
        # Prometheus Config
        self.test_file_exists("config/prometheus.yml", 6, "Monitoring")
        
        # Grafana Dashboards
        self.test_file_exists("observability/dashboards/ml_anomaly_dashboard.json", 6, "Dashboards")
        self.test_file_exists("observability/dashboards/request_latency_dashboard.json", 6, "Dashboards")
        self.test_file_exists("observability/dashboards/llm_cost_dashboard.json", 6, "Dashboards")
        
        # CI/CD
        self.test_file_exists(".github/workflows/ci.yml", 6, "CI/CD")
        self.test_file_exists(".github/workflows/docker.yml", 6, "CI/CD")
        
        # Documentation
        self.test_file_exists("docs/API.md", 6, "Documentation")
        self.test_file_exists("docs/DEVELOPER_SETUP.md", 6, "Documentation")
        
        # Load Testing
        self.test_file_exists("locustfile.py", 6, "Load Testing")
    
    # =========================================================================
    # Test Files Validation
    # =========================================================================
    
    def validate_tests(self):
        """Validate test coverage."""
        print("\n🧪 Test Files")
        print("-" * 50)
        
        test_files = [
            "tests/test_file_tools.py",
            "tests/test_git_tools.py",
            "tests/test_auth_gateway.py",
            "tests/test_intent_checker.py",
            "tests/test_anomaly_detector.py",
            "tests/test_ensemble_anomaly.py",
            "tests/test_hybrid_search.py",
            "tests/test_code_review_phase5.py",
            "tests/test_output_validator.py",
            "tests/test_model_training.py",
        ]
        
        for tf in test_files:
            self.test_file_exists(tf, 1, "Tests")
    
    # =========================================================================
    # Run All Validations
    # =========================================================================
    
    def run_all(self, phase_filter: Optional[int] = None):
        """Run all validations."""
        print("=" * 60)
        print("🔍 MCP Developer Assistant - Project Validation")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Project Root: {PROJECT_ROOT}")
        
        validations = [
            (0, self.validate_phase_0),
            (1, self.validate_phase_1),
            (2, self.validate_phase_2),
            (3, self.validate_phase_3),
            (4, self.validate_phase_4),
            (5, self.validate_phase_5),
            (6, self.validate_phase_6),
        ]
        
        for phase, validator in validations:
            if phase_filter is None or phase_filter == phase:
                try:
                    validator()
                except Exception as e:
                    print(f"  ❌ Error in Phase {phase}: {e}")
                    if self.verbose:
                        traceback.print_exc()
        
        self.validate_tests()
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for phase, report in sorted(self.phase_reports.items()):
            if report.total_tests == 0:
                continue
            
            status = "✅" if report.failed_tests == 0 else "⚠️" if report.pass_rate >= 80 else "❌"
            print(f"\n{status} Phase {phase}: {report.name}")
            print(f"   Passed: {report.passed_tests}/{report.total_tests} ({report.pass_rate:.1f}%)")
            
            total_passed += report.passed_tests
            total_failed += report.failed_tests
            
            # Show failed tests
            if report.failed_tests > 0 and self.verbose:
                for result in report.results:
                    if not result.passed:
                        print(f"   ❌ {result.name}: {result.message}")
                        if result.error:
                            print(f"      Error: {result.error[:100]}...")
        
        total = total_passed + total_failed
        overall_rate = (total_passed / total * 100) if total > 0 else 0
        
        print("\n" + "-" * 60)
        print(f"📈 OVERALL: {total_passed}/{total} tests passed ({overall_rate:.1f}%)")
        
        if overall_rate >= 95:
            print("🎉 Project is PRODUCTION READY!")
        elif overall_rate >= 80:
            print("👍 Project is MOSTLY COMPLETE - minor items pending")
        elif overall_rate >= 60:
            print("🔧 Project is IN PROGRESS - significant work remaining")
        else:
            print("🚧 Project needs substantial implementation")
        
        print("=" * 60)
        
        return overall_rate >= 80


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MCP Developer Assistant project")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--phase", "-p", type=int, help="Validate specific phase only")
    
    args = parser.parse_args()
    
    validator = ProjectValidator(verbose=args.verbose)
    success = validator.run_all(phase_filter=args.phase)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
