"""
Treuno 125M — Sandbox: Code Executor
Securely runs generated code in a subprocess sandbox with timeout enforcement.

Flow:
  1. Write code to a temp file with the correct extension
  2. Optionally compile (Rust, C, C++, Java)
  3. Run with subprocess + timeout
  4. Return stdout, stderr, exit_code, elapsed_time

Used by the verifier and inference engine for the self-correction loop.
"""

from __future__ import annotations
import os
import re
import time
import shutil
import logging
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from .languages import SUPPORTED_LANGUAGES, LanguageRunner, get_runner

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    language: str
    code: str
    stdout: str
    stderr: str
    exit_code: int
    elapsed_seconds: float
    timed_out: bool = False
    error_message: str = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def short_error(self) -> str:
        """Return a concise error for feeding back into the model."""
        if self.timed_out:
            return f"TimeoutError: Code exceeded {self.elapsed_seconds:.1f}s limit."
        if self.stderr.strip():
            # Return last 5 lines of stderr (most relevant for most languages)
            lines = self.stderr.strip().splitlines()
            return "\n".join(lines[-5:])
        return ""


class CodeExecutor:
    """
    Multi-language code execution sandbox using subprocess.

    Safety measures:
      - Strict timeout (default 10s, configurable per language)
      - Runs in a throwaway temp directory (deleted after execution)
      - No network access restrictions by default (add Docker for full isolation)
      - stdout/stderr length capped to prevent memory issues

    For production: wrap in Docker (--network=none --memory=256m --cpus=1)
    """

    def __init__(
        self,
        default_timeout: int = 10,
        max_output_chars: int = 8000,
    ):
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars

    def run(
        self,
        code: str,
        language: str,
        timeout: Optional[int] = None,
        stdin: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code in the specified language.

        Args:
            code:     Source code string
            language: Language identifier (e.g. "python", "javascript", "rust")
            timeout:  Override default timeout in seconds
            stdin:    Optional stdin to pipe into the program

        Returns:
            ExecutionResult with stdout, stderr, exit_code, elapsed_seconds
        """
        runner = get_runner(language)
        if runner is None:
            return ExecutionResult(
                language=language,
                code=code,
                stdout="",
                stderr=f"Unsupported language: {language}",
                exit_code=1,
                elapsed_seconds=0.0,
                error_message=f"Language '{language}' is not supported.",
            )

        t_out = timeout or runner.timeout or self.default_timeout
        work_dir = tempfile.mkdtemp(prefix="treuno_sandbox_")

        try:
            return self._execute(code, runner, work_dir, t_out, stdin)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _execute(
        self,
        code: str,
        runner: LanguageRunner,
        work_dir: str,
        timeout: int,
        stdin: Optional[str],
    ) -> ExecutionResult:
        # Write source file
        src_path = os.path.join(work_dir, f"main{runner.extension}")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Compile step (for Rust, C, C++, Java)
        compiled_path = None
        if runner.compile_cmd:
            out_path = os.path.join(work_dir, "main_out")
            compile_cmd = runner.get_compile_cmd(src_path, out_path)
            try:
                proc = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=work_dir,
                )
                if proc.returncode != 0:
                    return ExecutionResult(
                        language=runner.name,
                        code=code,
                        stdout="",
                        stderr=proc.stderr[:self.max_output_chars],
                        exit_code=proc.returncode,
                        elapsed_seconds=0.0,
                        error_message="Compilation failed.",
                    )
                compiled_path = out_path
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    language=runner.name,
                    code=code,
                    stdout="",
                    stderr="Compilation timed out.",
                    exit_code=1,
                    elapsed_seconds=float(timeout),
                    timed_out=True,
                )

        # Run step
        run_cmd = runner.get_run_cmd(src_path, compiled_path)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                run_cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )
            elapsed = time.perf_counter() - start
            return ExecutionResult(
                language=runner.name,
                code=code,
                stdout=proc.stdout[:self.max_output_chars],
                stderr=proc.stderr[:self.max_output_chars],
                exit_code=proc.returncode,
                elapsed_seconds=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return ExecutionResult(
                language=runner.name,
                code=code,
                stdout="",
                stderr=f"Execution timed out after {timeout}s.",
                exit_code=1,
                elapsed_seconds=elapsed,
                timed_out=True,
                error_message=f"Code exceeded {timeout}s execution limit.",
            )
        except FileNotFoundError as e:
            return ExecutionResult(
                language=runner.name,
                code=code,
                stdout="",
                stderr=str(e),
                exit_code=1,
                elapsed_seconds=0.0,
                error_message=f"Runtime not found: {e}. Is {runner.name} installed?",
            )

    @staticmethod
    def extract_code_block(text: str, language: Optional[str] = None) -> Optional[str]:
        """
        Extract the first code block from markdown-formatted model output.

        Args:
            text:     Model output string (may contain ```python ... ``` blocks)
            language: Expected language tag, or None to accept any

        Returns:
            Code string or None if no code block found
        """
        # Match ```language\n...\n``` or ```\n...\n```
        if language:
            pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
        else:
            pattern = r"```(?:\w+)?\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return whole text if it looks like raw code
        if text.strip() and not text.strip().startswith("I "):
            return text.strip()
        return None
