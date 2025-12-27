"""
AI-Powered Tools

LLM-enhanced tools with hybrid search, grounded answers, and validated outputs.

Enhanced Features:
- Hybrid search (BM25 + FAISS) with source citations
- Query expansion for better recall
- Validated LLM outputs with secret detection
- summarize_diff for PR descriptions
- Configurable detail levels
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from observability.logging_config import get_logger
from observability.metrics import metrics
from server.tools.base import BaseTool, ToolParameter, ToolResult

logger = get_logger(__name__)


class AskAboutCodeTool(BaseTool):
    """
    Ask questions about the codebase with semantic understanding.
    
    Enhanced with:
    - Hybrid search (BM25 + FAISS)
    - Source citations with line numbers
    - Query expansion for better recall
    - Validated, grounded responses
    """

    name = "ask_about_code"
    description = "Ask questions about the codebase using hybrid semantic search"
    parameters = [
        ToolParameter(name="query", description="Question about the code", type="string"),
        ToolParameter(name="top_k", description="Number of relevant code chunks", type="integer", required=False, default=5),
        ToolParameter(name="include_citations", description="Include source citations", type="boolean", required=False, default=True),
        ToolParameter(name="expand_query", description="Expand query with synonyms", type="boolean", required=False, default=True),
    ]

    def __init__(self, hybrid_search_engine=None, llm_client=None, embedding_manager=None):
        """
        Initialize tool.
        
        Args:
            hybrid_search_engine: HybridSearchEngine instance (preferred)
            llm_client: LLM client for generating answers
            embedding_manager: Legacy embedding manager (fallback)
        """
        super().__init__()
        self.hybrid_search = hybrid_search_engine
        self.embedding_manager = embedding_manager  # Fallback
        self.llm = llm_client

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        include_citations: bool = True,
        expand_query: bool = True,
    ) -> ToolResult:
        """Answer question about code with grounded sources."""
        if not self.llm:
            return ToolResult(
                success=False,
                result=None,
                error="LLM not initialized. Check configuration.",
            )

        if not self.hybrid_search and not self.embedding_manager:
            return ToolResult(
                success=False,
                result=None,
                error="No search engine available. Index the codebase first.",
            )

        try:
            # Expand query if enabled
            search_query = query
            if expand_query and self.hybrid_search:
                search_query = self.hybrid_search.expand_query(query)

            # Use hybrid search if available
            if self.hybrid_search:
                search_results = self.hybrid_search.search(search_query, top_k=top_k)
                context, sources = self._format_hybrid_results(search_results)
            else:
                # Fallback to legacy embedding manager
                query_embedding = self.embedding_manager.embed(query)
                relevant_files = self.embedding_manager.search(query_embedding, top_k=top_k)
                context, sources = await self._format_legacy_results(relevant_files)

            # Generate grounded answer
            prompt = self._build_qa_prompt(query, context, include_citations)
            response = await self.llm.generate(prompt)

            # Validate and parse response
            result = self._parse_response(response, sources, include_citations)

            logger.info(
                "ask_about_code",
                query=query[:50],
                sources=len(sources),
                expanded=expand_query,
            )

            return ToolResult(success=True, result=result)

        except Exception as e:
            logger.error("ask_about_code_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))

    def _format_hybrid_results(self, results: List[Dict]) -> tuple:
        """Format hybrid search results into context."""
        context_parts = []
        sources = []

        for result in results:
            file_path = result["file_path"]
            start_line = result["start_line"]
            end_line = result["end_line"]
            content = result["content"]

            header = f"--- {file_path} (lines {start_line}-{end_line}) ---"
            context_parts.append(f"{header}\n{content}")

            sources.append({
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "relevance_score": result.get("score", 0),
                "snippet": content[:200] if len(content) > 200 else content,
            })

        return "\n\n".join(context_parts), sources

    async def _format_legacy_results(self, file_paths: List[str]) -> tuple:
        """Format legacy search results."""
        from server.tools.file_tools import ReadFileTool
        file_tool = ReadFileTool()
        
        context_parts = []
        sources = []

        for file_path in file_paths:
            result = await file_tool.execute(file_path, max_lines=50)
            if result.success:
                content = result.result.get("content", "")
                context_parts.append(f"--- {file_path} ---\n{content}")
                sources.append({
                    "file_path": file_path,
                    "start_line": 1,
                    "end_line": result.result.get("lines", 50),
                })

        return "\n\n".join(context_parts), sources

    def _build_qa_prompt(self, query: str, context: str, include_citations: bool) -> str:
        """Build the QA prompt."""
        citation_instruction = ""
        if include_citations:
            citation_instruction = """
For each claim you make, cite the source using this format: [source: filename:line_range]
Example: "The function handles authentication [source: auth.py:45-60]"
"""

        return f"""Based ONLY on the following code snippets, answer this question.

Question: {query}

Code Context:
{context}

Requirements:
- Answer only based on the code snippets provided above
- If the information is not in the snippets, say "I couldn't find relevant code for this"
- Be specific and reference actual function/class names
{citation_instruction}
"""

    def _parse_response(self, response: str, sources: List[Dict], include_citations: bool) -> Dict:
        """Parse and structure the response."""
        result = {
            "query": "",
            "answer": response,
            "confidence": 0.8,  # Could be computed from search scores
        }

        if include_citations:
            result["sources"] = sources

        # Extract inline citations
        citation_pattern = r'\[source:\s*([^\]:]+):(\d+)-(\d+)\]'
        citations = re.findall(citation_pattern, response)
        if citations:
            result["inline_citations"] = [
                {"file": c[0], "start_line": int(c[1]), "end_line": int(c[2])}
                for c in citations
            ]

        return result


class ReviewChangesTool(BaseTool):
    """
    Review code changes with AI-powered risk assessment.
    
    Enhanced with:
    - ML-based risk scoring with feature extraction
    - Pydantic schema validation
    - Secret detection in responses
    - Security vulnerability detection
    - Review routing based on risk
    """

    name = "review_changes"
    description = "Review code changes with AI-powered risk assessment"
    parameters = [
        ToolParameter(name="diff", description="Git diff or commit ref", type="string"),
        ToolParameter(name="focus", description="Review focus: security, performance, style, all", type="string", required=False, default="all"),
        ToolParameter(name="repo_path", description="Repository path for git context", type="string", required=False, default="."),
        ToolParameter(name="author", description="Commit author for familiarity scoring", type="string", required=False),
    ]

    def __init__(self, llm_client=None, risk_model=None):
        """
        Initialize ReviewChangesTool.
        
        Args:
            llm_client: LLM client for generating reviews
            risk_model: Optional ML risk prediction model
        """
        super().__init__()
        self.llm = llm_client
        self._risk_scorer = None
        self._review_router = None
        self._risk_model = risk_model
    
    @property
    def risk_scorer(self):
        """Lazy-load risk scorer with ML model."""
        if self._risk_scorer is None:
            from ai.risk_scorer import RiskScorer
            self._risk_scorer = RiskScorer(ml_model=self._risk_model)
        return self._risk_scorer
    
    @property
    def review_router(self):
        """Lazy-load review router."""
        if self._review_router is None:
            from ai.review_router import ReviewRouter
            self._review_router = ReviewRouter()
        return self._review_router

    async def execute(
        self,
        diff: str,
        focus: str = "all",
        repo_path: str = ".",
        author: Optional[str] = None,
    ) -> ToolResult:
        """
        Review code changes with ML-enhanced risk scoring and validated output.
        
        Args:
            diff: Git diff content
            focus: Review focus area (security, performance, style, all)
            repo_path: Repository path for git history analysis
            author: Commit author for familiarity scoring
            
        Returns:
            ToolResult with review, risk score, and routing
        """
        if not self.llm:
            return ToolResult(
                success=False,
                result=None,
                error="LLM not initialized. Check configuration.",
            )

        try:
            # Import validators
            from ai.output_validator import (
                CodeReviewOutput,
                SecretScanner,
                validate_output,
                safe_parse_llm_response,
            )

            # Get ML-enhanced risk score
            risk_result = self.risk_scorer.score_with_features(diff, repo_path, author)
            
            # Truncate large diffs for LLM
            original_len = len(diff)
            work_diff = diff
            if len(diff) > 10000:
                work_diff = diff[:10000] + "\n... (truncated)"

            # Build enhanced prompt with risk context
            prompt = self._build_enhanced_review_prompt(work_diff, focus, risk_result)
            response = await self.llm.generate(prompt)

            # Scan response for secrets before returning
            scanner = SecretScanner(scan_pii=True)
            if scanner.has_secrets(response):
                response = scanner.redact(response)
                logger.warning("secrets_detected_in_review")

            # Validate output
            try:
                review = validate_output(response, CodeReviewOutput)
                result = review.model_dump()
            except Exception as e:
                # Fallback to basic JSON parsing
                logger.warning("review_validation_failed", error=str(e))
                result = self._parse_review_fallback(response)

            # Enrich result with ML risk scoring
            result = self._enrich_with_risk(result, risk_result)
            
            # Determine review routing
            routing = self.review_router.determine_route(
                risk_score=risk_result.score,
                risk_factors=risk_result.factors,
                security_patterns=list(result.get("security_concerns", [])),
                lines_changed=work_diff.count("\n+") + work_diff.count("\n-"),
                files_changed=len(re.findall(r"^\+\+\+ b/", work_diff, re.M)),
            )
            
            # Add routing to result
            result["routing"] = routing.to_dict()
            result["review_priority"] = routing.priority
            result["requires_security_review"] = routing.requires_security_review
            result["requires_senior_review"] = routing.requires_senior_review
            result["suggested_reviewers"] = routing.suggested_reviewers
            
            # Metadata
            result["diff_length"] = original_len
            result["truncated"] = original_len > 10000

            # Record metrics
            metrics.record_request("review_changes", 0)  # Latency tracked elsewhere
            
            logger.info(
                "review_changes",
                risk=result.get("risk_level"),
                ml_score=risk_result.ml_score,
                issues=len(result.get("issues", [])),
                priority=routing.priority,
            )
            
            return ToolResult(success=True, result=result)

        except Exception as e:
            logger.error("review_changes_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))
    
    def _build_enhanced_review_prompt(self, diff: str, focus: str, risk_result) -> str:
        """Build enhanced review prompt with risk context."""
        focus_instructions = {
            "security": "Focus on security vulnerabilities, injection risks, and authentication issues.",
            "performance": "Focus on performance issues, memory leaks, and inefficient algorithms.",
            "style": "Focus on code style, naming conventions, and readability.",
            "all": "Review for security, performance, style, and best practices.",
        }

        instruction = focus_instructions.get(focus, focus_instructions["all"])
        
        # Add risk context to prompt
        risk_context = ""
        if risk_result.factors:
            risk_context = f"""
Risk Assessment Context:
- Overall Risk Level: {risk_result.level}
- Risk Score: {risk_result.score:.2f}
- Key Risk Factors: {', '.join(risk_result.factors[:5])}

Pay special attention to the identified risk factors above.
"""

        return f"""Review this code diff. {instruction}

{risk_context}
Diff:
```diff
{diff}
```

Return your review as valid JSON matching this exact schema:
{{
  "summary": "One-line summary of changes (required, 10-500 chars)",
  "issues": [
    {{"file": "path/to/file.py", "line": 42, "severity": "HIGH|MEDIUM|LOW|INFO", "category": "security|performance|style|bug|maintainability|testing|other", "message": "Issue description", "suggestion": "How to fix"}}
  ],
  "test_suggestions": ["Test case descriptions"],
  "risk_level": "HIGH|MEDIUM|LOW",
  "estimated_review_time_minutes": 10,
  "security_concerns": ["Security issues if any"]
}}

Important: Do not include any actual secrets, API keys, or passwords in your response.
"""
    
    def _enrich_with_risk(self, result: Dict, risk_result) -> Dict:
        """Enrich review result with ML risk scoring data."""
        result["ml_risk_score"] = risk_result.ml_score
        result["ml_confidence"] = risk_result.ml_confidence
        result["risk_factors"] = risk_result.factors
        result["model_id"] = risk_result.model_id
        result["feature_contributions"] = risk_result.feature_contributions
        
        # Override risk_level if ML score is significantly different
        if risk_result.ml_score is not None:
            # Use ML-enhanced level if available
            if risk_result.level != result.get("risk_level"):
                logger.info(
                    "risk_level_adjusted",
                    llm_level=result.get("risk_level"),
                    ml_level=risk_result.level,
                )
            result["risk_level"] = risk_result.level
        
        return result

    def _parse_review_fallback(self, response: str) -> Dict:
        """Fallback JSON parsing."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "summary": "Review completed",
            "issues": [],
            "test_suggestions": [],
            "risk_level": "LOW",
            "raw_response": response[:1000],
        }


class SummarizeRepoTool(BaseTool):
    """
    Summarize repository structure and purpose.
    
    Enhanced with:
    - Configurable detail levels (brief/detailed)
    - Dependency analysis
    - Architecture detection
    """

    name = "summarize_repo"
    description = "Provide a high-level summary of the repository"
    parameters = [
        ToolParameter(name="path", description="Repository path", type="string", required=False, default="."),
        ToolParameter(name="detail_level", description="brief or detailed", type="string", required=False, default="brief"),
    ]

    def __init__(self, llm_client=None):
        super().__init__()
        self.llm = llm_client

    async def execute(self, path: str = ".", detail_level: str = "brief") -> ToolResult:
        """Summarize repository with configurable detail."""
        if not self.llm:
            return ToolResult(
                success=False,
                result=None,
                error="LLM not initialized. Check configuration.",
            )

        try:
            repo_path = Path(path)

            # Gather context
            context = self._gather_repo_context(repo_path, detail_level)

            # Build prompt
            prompt = self._build_summary_prompt(context, detail_level)
            response = await self.llm.generate(prompt)

            # Parse response
            result = self._parse_summary(response, path, detail_level)

            logger.info("summarize_repo", path=path, detail=detail_level)
            return ToolResult(success=True, result=result)

        except Exception as e:
            logger.error("summarize_repo_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))

    def _gather_repo_context(self, repo_path: Path, detail_level: str) -> Dict:
        """Gather repository context for summarization."""
        context = {
            "readme": "",
            "files": [],
            "dependencies": [],
            "config_files": [],
        }

        # Read README
        readme_paths = ["README.md", "README.rst", "README.txt", "README"]
        for readme_name in readme_paths:
            readme_path = repo_path / readme_name
            if readme_path.exists():
                max_len = 5000 if detail_level == "detailed" else 2000
                context["readme"] = readme_path.read_text(errors="replace")[:max_len]
                break

        # Gather file structure
        exclude = [".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"]
        file_limit = 200 if detail_level == "detailed" else 100

        for f in repo_path.rglob("*"):
            if f.is_file() and not any(p in str(f) for p in exclude):
                context["files"].append(str(f.relative_to(repo_path)))
                if len(context["files"]) >= file_limit:
                    break

        # Analyze dependencies
        dependency_files = {
            "requirements.txt": "python",
            "pyproject.toml": "python",
            "package.json": "javascript",
            "Cargo.toml": "rust",
            "go.mod": "go",
            "pom.xml": "java",
            "build.gradle": "java",
        }

        for dep_file, lang in dependency_files.items():
            dep_path = repo_path / dep_file
            if dep_path.exists():
                context["dependencies"].append({
                    "file": dep_file,
                    "language": lang,
                    "content": dep_path.read_text(errors="replace")[:1000] if detail_level == "detailed" else "",
                })

        # Config files
        config_files = ["docker-compose.yml", "Dockerfile", ".env.example", "config.yaml", "config.json"]
        for cfg in config_files:
            if (repo_path / cfg).exists():
                context["config_files"].append(cfg)

        return context

    def _build_summary_prompt(self, context: Dict, detail_level: str) -> str:
        """Build summary prompt."""
        if detail_level == "detailed":
            return f"""Provide a detailed summary of this code repository.

README:
{context['readme']}

File Structure ({len(context['files'])} files):
{chr(10).join(context['files'][:100])}

Dependencies:
{json.dumps(context['dependencies'], indent=2)}

Config Files: {', '.join(context['config_files'])}

Provide a detailed summary including:
1. Project purpose and goals (3-5 sentences)
2. Technologies and frameworks used (with versions if visible)
3. Architecture overview (components, layers, patterns)
4. Key modules and their responsibilities
5. How to set up and run the project
6. Notable design decisions or patterns

Return as JSON:
{{"purpose": "...", "technologies": [...], "key_components": [...], "getting_started": "...", "architecture": "..."}}
"""
        else:
            return f"""Summarize this code repository briefly.

README:
{context['readme']}

File Structure ({len(context['files'])} files):
{chr(10).join(context['files'][:50])}

Dependencies detected: {[d['file'] for d in context['dependencies']]}

Provide a brief summary:
1. Project purpose (1-2 sentences)
2. Main technologies
3. Key components
4. Quick start instructions

Return as JSON:
{{"purpose": "...", "technologies": [...], "key_components": [...], "getting_started": "..."}}
"""

    def _parse_summary(self, response: str, path: str, detail_level: str) -> Dict:
        """Parse summary response."""
        result = {"path": path, "detail_level": detail_level}

        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                result.update(parsed)
                return result
            except json.JSONDecodeError:
                pass

        # Fallback to raw response
        result["summary"] = response
        return result


class SummarizeDiffTool(BaseTool):
    """
    Summarize git diff for PR descriptions.
    
    Generates clear, human-readable summaries suitable for
    commit messages and pull request descriptions.
    """

    name = "summarize_diff"
    description = "Generate a PR/commit summary from a git diff"
    parameters = [
        ToolParameter(name="diff", description="Git diff content", type="string"),
        ToolParameter(name="detail_level", description="brief or detailed", type="string", required=False, default="brief"),
        ToolParameter(name="format", description="Output format: text, markdown, json", type="string", required=False, default="markdown"),
    ]

    def __init__(self, llm_client=None):
        super().__init__()
        self.llm = llm_client

    async def execute(
        self,
        diff: str,
        detail_level: str = "brief",
        format: str = "markdown",
    ) -> ToolResult:
        """Summarize diff for PR description."""
        if not self.llm:
            return ToolResult(
                success=False,
                result=None,
                error="LLM not initialized. Check configuration.",
            )

        try:
            # Analyze diff metadata
            diff_stats = self._analyze_diff(diff)

            # Truncate if needed
            work_diff = diff
            if len(diff) > 15000:
                work_diff = diff[:15000] + "\n... (truncated)"

            # Build prompt
            prompt = self._build_diff_prompt(work_diff, detail_level, format)
            response = await self.llm.generate(prompt)

            # Parse response
            result = self._parse_diff_summary(response, diff_stats, format)

            logger.info(
                "summarize_diff",
                files=diff_stats["files_affected"],
                additions=diff_stats["lines_added"],
            )

            return ToolResult(success=True, result=result)

        except Exception as e:
            logger.error("summarize_diff_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))

    def _analyze_diff(self, diff: str) -> Dict:
        """Extract statistics from diff."""
        lines = diff.split("\n")

        stats = {
            "files_affected": 0,
            "lines_added": 0,
            "lines_removed": 0,
            "files": [],
        }

        current_file = None
        for line in lines:
            if line.startswith("diff --git"):
                stats["files_affected"] += 1
                # Extract filename
                match = re.search(r'b/(.+)$', line)
                if match:
                    current_file = match.group(1)
                    stats["files"].append(current_file)
            elif line.startswith("+") and not line.startswith("+++"):
                stats["lines_added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                stats["lines_removed"] += 1

        return stats

    def _build_diff_prompt(self, diff: str, detail_level: str, format: str) -> str:
        """Build diff summary prompt."""
        format_instructions = {
            "text": "Return as plain text suitable for a commit message.",
            "markdown": "Return as markdown with sections for title, description, and changes.",
            "json": 'Return as JSON: {"title": "...", "description": "...", "changes": [...], "breaking_changes": [...]}'
        }

        detail_instructions = {
            "brief": "Keep the summary concise (1 paragraph, 3-5 bullet points).",
            "detailed": "Provide detailed explanation of each change with rationale.",
        }

        return f"""Analyze this git diff and create a {detail_level} summary suitable for a PR description.

{diff}

{detail_instructions.get(detail_level, detail_instructions['brief'])}
{format_instructions.get(format, format_instructions['markdown'])}

Guidelines:
- Start with a clear one-line title
- Explain WHAT changed and WHY
- List any breaking changes prominently
- Note any new dependencies or configuration changes
- Keep technical details that reviewers need to know
"""

    def _parse_diff_summary(self, response: str, stats: Dict, format: str) -> Dict:
        """Parse diff summary response."""
        result = {
            "files_affected": stats["files_affected"],
            "lines_added": stats["lines_added"],
            "lines_removed": stats["lines_removed"],
            "files": stats["files"][:20],  # Limit
        }

        if format == "json":
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    result.update(parsed)
                    return result
                except json.JSONDecodeError:
                    pass

        result["summary"] = response
        return result


class QueryExpansionTool(BaseTool):
    """
    Expand search queries with synonyms and related terms.
    
    Improves recall for semantic code search.
    """

    name = "expand_query"
    description = "Expand a search query with synonyms for better code search"
    parameters = [
        ToolParameter(name="query", description="Original search query", type="string"),
    ]

    # Code-specific synonyms
    SYNONYMS = {
        # Python/general
        "function": ["func", "method", "def", "procedure", "subroutine"],
        "class": ["type", "struct", "interface", "model", "entity"],
        "error": ["exception", "error", "fault", "bug", "issue", "failure"],
        "test": ["test", "spec", "check", "verify", "assert"],
        "config": ["configuration", "settings", "options", "preferences", "params"],
        
        # Web/API
        "auth": ["authentication", "authorization", "login", "signin", "oauth"],
        "api": ["endpoint", "route", "handler", "controller", "resource"],
        "request": ["req", "http", "fetch", "call", "invoke"],
        "response": ["res", "reply", "result", "output", "return"],
        
        # Data
        "db": ["database", "sql", "query", "storage", "persistence"],
        "cache": ["cache", "memo", "store", "buffer"],
        "async": ["async", "await", "promise", "future", "concurrent"],
        
        # Security
        "secret": ["secret", "key", "credential", "password", "token"],
        "encrypt": ["encrypt", "cipher", "hash", "secure", "protect"],
        "validate": ["validate", "verify", "check", "sanitize", "filter"],
    }

    def __init__(self):
        super().__init__()

    async def execute(self, query: str) -> ToolResult:
        """Expand query with synonyms."""
        try:
            expanded_terms = set()
            query_lower = query.lower()
            
            # Find matching synonyms
            for term, synonyms in self.SYNONYMS.items():
                if term in query_lower:
                    expanded_terms.update(synonyms)
            
            # Combine with original query
            expanded_query = query
            if expanded_terms:
                expanded_query = f"{query} {' '.join(expanded_terms)}"
            
            result = {
                "original_query": query,
                "expanded_query": expanded_query,
                "added_terms": list(expanded_terms),
            }
            
            return ToolResult(success=True, result=result)
            
        except Exception as e:
            logger.error("expand_query_error", error=str(e))
            return ToolResult(success=False, result=None, error=str(e))
