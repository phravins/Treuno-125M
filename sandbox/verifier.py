"""
Treuno 125M — Sandbox: Code Verifier
Self-correction loop: run generated code → check result → feed errors back.

The verifier is the final quality gate before Treuno returns an answer.
If code fails, the error is injected back into the prompt for regeneration,
up to max_retries attempts. This eliminates hallucinated APIs and syntax errors
without any explicit error-detection logic — just run it and see.

Usage:
    verifier = CodeVerifier()
    result = verifier.verify("print(2 + 2)", "python", expected_output="4")
    if result.passed:
        print("Code is correct!")
    else:
        print(result.feedback_prompt)  # Feed this back to the model
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from .executor import CodeExecutor, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    passed: bool
    execution: ExecutionResult
    expected_output: Optional[str]
    feedback_prompt: str = ""    # Prompt to append for self-correction
    attempt: int = 1

    @property
    def stdout(self) -> str:
        return self.execution.stdout

    @property
    def stderr(self) -> str:
        return self.execution.stderr

    @property
    def exit_code(self) -> int:
        return self.execution.exit_code


class CodeVerifier:
    """
    Runs generated code and determines if it's correct.

    Correctness criteria:
      1. Exit code == 0 (no crash)
      2. If expected_output is given: stdout matches (stripped, normalized)
      3. No timeout

    On failure, generates a feedback_prompt string containing the
    error details — ready to be appended to the original prompt for
    re-generation by TreunoModel.
    """

    def __init__(self, default_timeout: int = 10):
        self.executor = CodeExecutor(default_timeout=default_timeout)

    def verify(
        self,
        code: str,
        language: str,
        expected_output: Optional[str] = None,
        timeout: Optional[int] = None,
        stdin: Optional[str] = None,
    ) -> VerificationResult:
        """
        Execute code and verify its correctness.

        Args:
            code:            Source code to verify
            language:        Language identifier
            expected_output: If given, stdout must match this (stripped)
            timeout:         Execution timeout in seconds
            stdin:           Optional stdin for the program

        Returns:
            VerificationResult with .passed, .feedback_prompt
        """
        exec_result = self.executor.run(code, language, timeout=timeout, stdin=stdin)

        # ── Check exit code ──────────────────────────────────────────────────
        if not exec_result.success:
            feedback = self._build_error_feedback(code, exec_result)
            return VerificationResult(
                passed=False,
                execution=exec_result,
                expected_output=expected_output,
                feedback_prompt=feedback,
            )

        # ── Check expected output ────────────────────────────────────────────
        if expected_output is not None:
            actual = exec_result.stdout.strip()
            expected = expected_output.strip()
            if actual != expected:
                feedback = self._build_output_mismatch_feedback(
                    code, expected, actual
                )
                return VerificationResult(
                    passed=False,
                    execution=exec_result,
                    expected_output=expected_output,
                    feedback_prompt=feedback,
                )

        return VerificationResult(
            passed=True,
            execution=exec_result,
            expected_output=expected_output,
            feedback_prompt="",
        )

    def verify_with_retries(
        self,
        code: str,
        language: str,
        generate_fn,
        original_prompt: str,
        max_retries: int = 3,
        expected_output: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> VerificationResult:
        """
        Run verify → on failure, call generate_fn(prompt + error) → retry.

        Args:
            code:            Initial generated code
            language:        Language identifier
            generate_fn:     Callable(prompt: str) -> str (model generate)
            original_prompt: The user's original prompt
            max_retries:     Max self-correction attempts
            expected_output: Optional expected stdout
            timeout:         Execution timeout

        Returns:
            Final VerificationResult (passed or exhausted retries)
        """
        current_code = code
        for attempt in range(1, max_retries + 1):
            result = self.verify(current_code, language, expected_output, timeout)
            result.attempt = attempt

            if result.passed:
                logger.info(f"Code verified on attempt {attempt}.")
                return result

            if attempt == max_retries:
                logger.warning(f"Code failed after {max_retries} attempts.")
                return result

            # Build correction prompt and regenerate
            correction_prompt = (
                f"{original_prompt}\n\n"
                f"Your previous code attempt failed:\n"
                f"```{language}\n{current_code}\n```\n\n"
                f"Error:\n{result.feedback_prompt}\n\n"
                f"Please fix the code and try again."
            )
            logger.info(f"Self-correcting (attempt {attempt + 1}/{max_retries})...")
            raw_output = generate_fn(correction_prompt)
            new_code = self.executor.extract_code_block(raw_output, language)
            if new_code:
                current_code = new_code
            else:
                current_code = raw_output  # Use raw output if no code block found

        return result  # type: ignore

    @staticmethod
    def _build_error_feedback(code: str, exec_result: ExecutionResult) -> str:
        lines = [
            f"The following {exec_result.language} code produced an error:",
            f"```\n{code}\n```",
        ]
        if exec_result.timed_out:
            lines.append(f"Error: Execution timed out after {exec_result.elapsed_seconds:.1f}s.")
        elif exec_result.stderr.strip():
            lines.append(f"Error output:\n{exec_result.short_error}")
        lines.append("Please fix the code.")
        return "\n".join(lines)

    @staticmethod
    def _build_output_mismatch_feedback(
        code: str, expected: str, actual: str
    ) -> str:
        return (
            f"The code ran without errors but produced incorrect output.\n"
            f"Expected: {repr(expected)}\n"
            f"Got:      {repr(actual)}\n"
            f"Please fix the code to produce the correct output."
        )
