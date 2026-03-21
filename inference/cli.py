"""
Treuno 125M — Interactive CLI
A rich terminal REPL for querying the model locally.

Usage:
    python inference/cli.py
    python inference/cli.py --no-retrieval --no-sandbox
    python inference/cli.py --model-path d:/MODEL/weights
"""

from __future__ import annotations
import os
import sys
import textwrap
import argparse
import logging

# Ensure the project root (d:/MODEL) is on sys.path regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

# ── Redirect HuggingFace Cache ───────────────────────────────────────────────
# The D: drive has more space than C:, and GPT-2 weights are ~550MB.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(ROOT, ".cache", "huggingface")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="treuno",
        description="Treuno 125M — Coding LM with live retrieval and code verification",
    )
    p.add_argument("--model-path", default="d:/MODEL/weights",
                   help="Path to model weights directory")
    p.add_argument("--no-retrieval", action="store_true",
                   help="Disable Antigravity web retrieval")
    p.add_argument("--no-sandbox", action="store_true",
                   help="Disable code execution sandbox")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--language", default=None,
                   help="Force output language (e.g. python, javascript)")
    return p


def print_banner():
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]Treuno 125M[/bold cyan]\n"
            "[dim]Coding LM · Antigravity Retrieval · Code Sandbox[/dim]\n"
            "[dim]Type your coding question. Type [bold]/exit[/bold] to quit.[/dim]",
            border_style="cyan",
        ))
    except ImportError:
        print("=" * 55)
        print("  Treuno 125M — Coding LM with Antigravity + Sandbox")
        print("  Type your question. Type /exit to quit.")
        print("=" * 55)


def print_result(result, use_rich: bool = True):
    try:
        if use_rich:
            from rich.console import Console
            from rich.syntax import Syntax
            from rich.panel import Panel
            from rich.markdown import Markdown
            console = Console()

            if result.retrieved_sources:
                console.print(f"\n[dim]📡 Antigravity: {len(result.retrieved_sources)} sources retrieved[/dim]")

            console.print(Markdown(result.text))

            if result.code and result.language:
                console.print(Panel(
                    Syntax(result.code, result.language, theme="monokai", line_numbers=True),
                    title=f"[green]{result.language}[/green]",
                    border_style="green" if result.verified else "yellow",
                ))

            if result.verified:
                console.print(f"[green]✅ Code verified[/green] (attempt {result.verification.attempt})")
                if result.stdout.strip():
                    console.print(Panel(result.stdout.strip(), title="Output", border_style="dim"))
            elif result.verification:
                console.print(f"[yellow]⚠️  Code could not be verified after {result.verification.attempt} attempts[/yellow]")
            return
    except ImportError:
        pass

    # Fallback: plain text
    print("\n" + result.text)
    if result.code:
        print(f"\n--- {result.language or 'code'} ---")
        print(result.code)
    if result.verified:
        print(f"\n✅ Verified")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    print_banner()
    print()

    from inference.engine import TreunoEngine
    print("Loading Treuno engine...")
    engine = TreunoEngine.from_pretrained(
        args.model_path,
        use_retrieval=not args.no_retrieval,
        use_sandbox=not args.no_sandbox,
    )

    retrieval_status = "off" if args.no_retrieval else "on"
    sandbox_status = "off" if args.no_sandbox else "on"
    print(f"Ready! (retrieval={retrieval_status}, sandbox={sandbox_status})\n")

    while True:
        try:
            query = input("treuno> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye!")
            break
        if query == "/help":
            print(textwrap.dedent("""
              Commands:
                /exit          Quit
                /help          Show this help
                /info          Show model info

              Just type any coding question to generate verified code.
            """))
            continue
        if query == "/info":
            from model.config import TreunoConfig
            cfg = TreunoConfig.treuno_125m()
            print(f"Model: {cfg.model_name} v{cfg.model_version}")
            print(f"Params: {cfg.param_estimate():,}")
            print(f"Context: {cfg.context_length:,} tokens")
            continue

        try:
            result = engine.generate(
                prompt=query,
                language=args.language,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print_result(result)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
