"""Base agent class for multi-agent data assistant.

This module provides the abstract base class for all agents in the
analyst loop. Each agent must implement the run() method and return
a structured output conforming to its contract.
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from haikugraph.agents.contracts import AgentStatus, AgentTrace
from haikugraph.llm.router import call_llm


# Type variable for agent output
T = TypeVar("T", bound=BaseModel)


class AgentError(Exception):
    """Exception raised when agent execution fails."""
    
    def __init__(self, message: str, agent_name: str, details: dict | None = None):
        super().__init__(message)
        self.agent_name = agent_name
        self.details = details or {}


class BaseAgent(ABC, Generic[T]):
    """Abstract base class for all agents.
    
    Each agent:
    1. Receives input from the agentic runtime
    2. Performs its task (possibly calling LLM)
    3. Returns a structured output conforming to its contract
    
    Subclasses must implement:
    - run(): Main execution method
    - output_schema: The Pydantic model for output validation
    """
    
    # Class-level configuration
    name: str = "base_agent"
    max_retries: int = 2
    llm_role: str = "planner"  # Role for LLM router
    
    def __init__(self, *, max_retries: int | None = None):
        """Initialize base agent.
        
        Args:
            max_retries: Override default max retries
        """
        if max_retries is not None:
            self.max_retries = max_retries
        
        self._start_time: float | None = None
        self._end_time: float | None = None
        self.llm_provider: str | None = None
        self.llm_model_overrides: dict[str, str] = {}
    
    @property
    @abstractmethod
    def output_schema(self) -> type[T]:
        """Return the Pydantic model class for output validation."""
        ...
    
    @abstractmethod
    def run(self, **kwargs) -> T:
        """Execute the agent's task.
        
        Args:
            **kwargs: Agent-specific input parameters
        
        Returns:
            Validated output conforming to output_schema
        """
        ...
    
    def _start_timer(self) -> None:
        """Start execution timer."""
        self._start_time = time.perf_counter()
    
    def _stop_timer(self) -> float:
        """Stop execution timer and return elapsed time in ms."""
        self._end_time = time.perf_counter()
        if self._start_time is None:
            return 0.0
        return (self._end_time - self._start_time) * 1000
    
    def _get_elapsed_ms(self) -> float:
        """Get elapsed time in ms without stopping timer."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000
    
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: str | None = None,
        max_tokens: int | None = None,
        timeout: int = 30,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """Call LLM with system and user prompts.
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            role: Override LLM role (default: self.llm_role)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        
        Returns:
            LLM response text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_llm(
            messages,
            role=role or self.llm_role,
            max_tokens=max_tokens,
            timeout=timeout,
            provider=provider or self.llm_provider,
            model=model or self.llm_model_overrides.get(role or self.llm_role),
        )
    
    def parse_json_response(
        self,
        response: str,
        schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Parse JSON from LLM response.
        
        Args:
            response: Raw LLM response text
            schema: Optional Pydantic model for validation
        
        Returns:
            Parsed JSON dict
        
        Raises:
            ValueError: If response is not valid JSON
            ValidationError: If schema validation fails
        """
        # Strip markdown code blocks
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove ```json line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from text
            import re
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse JSON: {e}") from e
            else:
                raise ValueError(f"Failed to parse JSON: {e}") from e
        
        # Validate against schema if provided
        if schema is not None:
            schema(**data)  # This will raise ValidationError if invalid
        
        return data
    
    def validate_output(self, data: dict[str, Any]) -> T:
        """Validate output against the agent's schema.
        
        Args:
            data: Dictionary to validate
        
        Returns:
            Validated Pydantic model instance
        
        Raises:
            ValidationError: If validation fails
        """
        return self.output_schema(**data)
    
    def create_trace(
        self,
        status: AgentStatus,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AgentTrace:
        """Create an execution trace for this agent.
        
        Args:
            status: Execution status
            input_summary: Summary of input (optional)
            output_summary: Summary of output (optional)
            error: Error message if failed (optional)
        
        Returns:
            AgentTrace instance
        """
        now = datetime.utcnow()
        duration_ms = self._get_elapsed_ms()
        
        return AgentTrace(
            agent_name=self.name,
            started_at=now,  # Approximation
            completed_at=now,
            duration_ms=duration_ms,
            status=status,
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            error=error,
        )
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"


class LLMAgent(BaseAgent[T], ABC):
    """Base class for agents that primarily use LLM for their task.
    
    Provides additional utilities for LLM-based agents including:
    - Retry logic for LLM calls
    - JSON repair prompts
    - Schema-guided generation
    """
    
    # Override in subclass
    system_prompt: str = ""
    user_prompt_template: str = ""
    repair_prompt_template: str = """The previous JSON output had errors. Fix it.

Previous output:
{previous_output}

Errors:
{errors}

Rules:
- Output ONLY valid JSON (no markdown)
- Follow the exact schema required
- No extra fields

Return ONLY the corrected JSON."""
    
    def call_llm_with_retry(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        expected_schema: type[BaseModel] | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Call LLM with retry and JSON repair logic.
        
        Args:
            user_prompt: User prompt text
            system_prompt: Override system prompt
            expected_schema: Schema to validate against
            max_retries: Override max retries
        
        Returns:
            Parsed and validated JSON dict
        
        Raises:
            AgentError: If all retries fail
        """
        retries = max_retries if max_retries is not None else self.max_retries
        sys_prompt = system_prompt or self.system_prompt
        
        last_error = None
        last_response = ""
        
        for attempt in range(retries + 1):
            try:
                if attempt == 0:
                    response = self.call_llm(sys_prompt, user_prompt)
                else:
                    # Repair attempt
                    repair_prompt = self.repair_prompt_template.format(
                        previous_output=last_response,
                        errors=str(last_error),
                    )
                    response = self.call_llm(sys_prompt, repair_prompt)
                
                last_response = response
                
                # Parse and validate
                data = self.parse_json_response(response)
                
                if expected_schema:
                    expected_schema(**data)
                
                return data
            
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_error = e
                continue
        
        raise AgentError(
            f"LLM call failed after {retries} retries: {last_error}",
            agent_name=self.name,
            details={"last_response": last_response, "error": str(last_error)},
        )
