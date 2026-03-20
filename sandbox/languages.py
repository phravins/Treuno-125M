"""
Treuno 125M — Sandbox: Language Runners
Per-language strategies for executing code in the sandbox.
Each runner knows the command, file extension, and compilation step (if any).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LanguageRunner:
    name: str
    extension: str         # File extension for temp file
    run_cmd: List[str]     # Command template; use {file} as placeholder
    compile_cmd: Optional[List[str]] = None   # Optional compile step
    version_cmd: List[str] = None            # e.g. ["python", "--version"]
    timeout: int = 10                        # Default execution timeout in seconds

    def get_run_cmd(self, filepath: str, compiled_path: Optional[str] = None) -> List[str]:
        """Fill in the {file} placeholder with actual filepath."""
        target = compiled_path or filepath
        return [part.replace("{file}", target) for part in self.run_cmd]

    def get_compile_cmd(self, filepath: str, output_path: str) -> Optional[List[str]]:
        if not self.compile_cmd:
            return None
        return [
            part.replace("{file}", filepath).replace("{out}", output_path)
            for part in self.compile_cmd
        ]


# ── Supported language definitions ───────────────────────────────────────────

SUPPORTED_LANGUAGES: Dict[str, LanguageRunner] = {
    "python": LanguageRunner(
        name="Python",
        extension=".py",
        run_cmd=["python", "{file}"],
        version_cmd=["python", "--version"],
        timeout=10,
    ),
    "javascript": LanguageRunner(
        name="JavaScript",
        extension=".js",
        run_cmd=["node", "{file}"],
        version_cmd=["node", "--version"],
        timeout=10,
    ),
    "typescript": LanguageRunner(
        name="TypeScript",
        extension=".ts",
        run_cmd=["ts-node", "{file}"],
        version_cmd=["ts-node", "--version"],
        timeout=15,
    ),
    "bash": LanguageRunner(
        name="Bash",
        extension=".sh",
        run_cmd=["bash", "{file}"],
        version_cmd=["bash", "--version"],
        timeout=10,
    ),
    "rust": LanguageRunner(
        name="Rust",
        extension=".rs",
        run_cmd=["{file}"],
        compile_cmd=["rustc", "{file}", "-o", "{out}"],
        version_cmd=["rustc", "--version"],
        timeout=30,
    ),
    "go": LanguageRunner(
        name="Go",
        extension=".go",
        run_cmd=["go", "run", "{file}"],
        version_cmd=["go", "version"],
        timeout=15,
    ),
    "c": LanguageRunner(
        name="C",
        extension=".c",
        run_cmd=["{file}"],
        compile_cmd=["gcc", "{file}", "-o", "{out}"],
        version_cmd=["gcc", "--version"],
        timeout=20,
    ),
    "cpp": LanguageRunner(
        name="C++",
        extension=".cpp",
        run_cmd=["{file}"],
        compile_cmd=["g++", "{file}", "-o", "{out}"],
        version_cmd=["g++", "--version"],
        timeout=20,
    ),
    "java": LanguageRunner(
        name="Java",
        extension=".java",
        run_cmd=["java", "{file}"],
        compile_cmd=["javac", "{file}"],
        version_cmd=["java", "--version"],
        timeout=20,
    ),
    "ruby": LanguageRunner(
        name="Ruby",
        extension=".rb",
        run_cmd=["ruby", "{file}"],
        version_cmd=["ruby", "--version"],
        timeout=10,
    ),
    "php": LanguageRunner(
        name="PHP",
        extension=".php",
        run_cmd=["php", "{file}"],
        version_cmd=["php", "--version"],
        timeout=10,
    ),
    "swift": LanguageRunner(
        name="Swift",
        extension=".swift",
        run_cmd=["swift", "{file}"],
        version_cmd=["swift", "--version"],
        timeout=20,
    ),
    "kotlin": LanguageRunner(
        name="Kotlin",
        extension=".kts",
        run_cmd=["kotlinc", "-script", "{file}"],
        version_cmd=["kotlinc", "-version"],
        timeout=30,
    ),
    "r": LanguageRunner(
        name="R",
        extension=".r",
        run_cmd=["Rscript", "{file}"],
        version_cmd=["Rscript", "--version"],
        timeout=15,
    ),
}

# Aliases
SUPPORTED_LANGUAGES["js"] = SUPPORTED_LANGUAGES["javascript"]
SUPPORTED_LANGUAGES["ts"] = SUPPORTED_LANGUAGES["typescript"]
SUPPORTED_LANGUAGES["sh"] = SUPPORTED_LANGUAGES["bash"]
SUPPORTED_LANGUAGES["c++"] = SUPPORTED_LANGUAGES["cpp"]


def get_runner(language: str) -> Optional[LanguageRunner]:
    """Get the LanguageRunner for a given language string (case-insensitive)."""
    return SUPPORTED_LANGUAGES.get(language.lower().strip())
