import difflib
import json
import subprocess
import tempfile
import textwrap
from typing import List, Dict, Optional

import torch

from model_client import (
    ModelLoadConfig,
    ModelSelector,
)


class CodingAgentEvaluator:
    """
    Evaluates whether a fine-tuned model produces better code patches
    than its base model counterpart, using:
      - Diff similarity score  (always)
      - Syntax validity check  (always)
      - Unit test pass rate     (optional, when test_code is provided)
    """

    def __init__(self, config: ModelLoadConfig):
        self.selector = ModelSelector(config)

        print("Loading base model...")
        self.base_model, self.tokenizer = self.selector.load_base()

        print("Loading fine-tuned model...")
        self.ft_model, _ = self.selector.load_finetuned()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_patch(
        self,
        model,
        instruction: str,
        source_code: str,
        max_tokens: int = 512,
    ) -> str:
        """Run a single forward pass and return the model's raw text."""

        prompt = f"""You are an expert software engineer.

Task:
{instruction}

Source Code:
```python
{source_code}
```

Return ONLY the modified Python code, with no explanation."""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )
        return response

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def similarity_score(self, generated: str, expected: str) -> float:
        """Normalised edit-distance similarity in [0, 1]."""
        return difflib.SequenceMatcher(None, generated, expected).ratio()

    def syntax_valid(self, code: str) -> bool:
        """Return True if the code string parses without a SyntaxError."""
        try:
            compile(textwrap.dedent(code), "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def run_tests(self, code: str, test_code: str) -> float:
        """
        Write `code` + `test_code` to a temp file, run pytest, and
        return the fraction of tests that passed.  Returns 0.0 on
        any execution error.
        """
        combined = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)

        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(combined)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["pytest", tmp_path, "--tb=no", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # pytest summary line: "2 passed, 1 failed in 0.12s"
            passed = failed = 0
            for token in result.stdout.split():
                if token.isdigit():
                    # first number = passed, second = failed (if present)
                    if passed == 0:
                        passed = int(token)
                    else:
                        failed = int(token)
                        break
            total = passed + failed
            return passed / total if total > 0 else 0.0
        except Exception as e:
            print(f"  [test runner error] {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Per-case evaluation
    # ------------------------------------------------------------------

    def evaluate_case(self, case: Dict) -> Dict:
        """
        Evaluate one test case.  `case` must contain:
          - instruction  : str
          - source_code  : str
          - expected     : str

        Optionally:
          - test_code    : str  (pytest-style tests to run against the output)
        """
        print("=" * 80)
        print(f"TASK : {case['instruction']}")

        base_output = self.generate_patch(
            self.base_model,
            case["instruction"],
            case["source_code"],
        )
        ft_output = self.generate_patch(
            self.ft_model,
            case["instruction"],
            case["source_code"],
        )

        expected = case["expected"]
        test_code: Optional[str] = case.get("test_code")

        # --- Similarity ---
        base_sim = self.similarity_score(base_output, expected)
        ft_sim   = self.similarity_score(ft_output,   expected)

        # --- Syntax validity ---
        base_syntax = self.syntax_valid(base_output)
        ft_syntax   = self.syntax_valid(ft_output)

        # --- Unit tests (optional) ---
        base_tests = ft_tests = None
        if test_code:
            base_tests = self.run_tests(base_output, test_code)
            ft_tests   = self.run_tests(ft_output,   test_code)

        # --- Composite score ---
        # Weights: similarity 50 %, syntax 20 %, test pass rate 30 %
        def composite(sim, syntax, tests):
            score = sim * 0.5 + (1.0 if syntax else 0.0) * 0.2
            if tests is not None:
                score += tests * 0.3
            else:
                # re-normalise without test component
                score = sim * 0.7 + (1.0 if syntax else 0.0) * 0.3
            return score

        base_composite = composite(base_sim, base_syntax, base_tests)
        ft_composite   = composite(ft_sim,   ft_syntax,   ft_tests)

        winner = "finetuned" if ft_composite > base_composite else "base"

        # --- Logging ---
        print(f"  Similarity  — base: {base_sim:.4f}  |  finetuned: {ft_sim:.4f}")
        print(f"  Syntax OK   — base: {base_syntax}      |  finetuned: {ft_syntax}")
        if test_code:
            print(f"  Test pass   — base: {base_tests:.2%}   |  finetuned: {ft_tests:.2%}")
        print(f"  Composite   — base: {base_composite:.4f}  |  finetuned: {ft_composite:.4f}")
        print(f"  Winner      : {winner.upper()}")

        return {
            "instruction":    case["instruction"],
            "winner":         winner,
            "base_sim":       base_sim,
            "ft_sim":         ft_sim,
            "base_syntax":    base_syntax,
            "ft_syntax":      ft_syntax,
            "base_tests":     base_tests,
            "ft_tests":       ft_tests,
            "base_composite": base_composite,
            "ft_composite":   ft_composite,
        }

    # ------------------------------------------------------------------
    # Full evaluation loop
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_cases: List[Dict],
        save_results: Optional[str] = None,
    ) -> List[Dict]:
        """
        Run all test cases and print a final leaderboard.

        Args:
            test_cases:    List of case dicts (see evaluate_case).
            save_results:  Optional filepath to dump JSON results.

        Returns:
            List of per-case result dicts.
        """
        results = []
        base_wins = ft_wins = 0

        for case in test_cases:
            result = self.evaluate_case(case)
            results.append(result)
            if result["winner"] == "finetuned":
                ft_wins += 1
            else:
                base_wins += 1

        # ---- Summary ----
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"  Base wins      : {base_wins}")
        print(f"  Finetuned wins : {ft_wins}")

        avg_base_sim = sum(r["base_sim"] for r in results) / len(results)
        avg_ft_sim   = sum(r["ft_sim"]   for r in results) / len(results)
        print(f"  Avg similarity — base: {avg_base_sim:.4f}  |  finetuned: {avg_ft_sim:.4f}")

        if ft_wins > base_wins:
            print("\n  Fine-tuned model is BETTER")
        elif base_wins > ft_wins:
            print("\n  Base model is BETTER")
        else:
            print("\n  It's a DRAW")

        if save_results:
            with open(save_results, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Results saved to {save_results}")

        return results


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":

    config = ModelLoadConfig(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",       # e.g. "meta-llama/Llama-3-8B"
        adapter_path="./adapter",           # path to your LoRA adapter
    )

    evaluator = CodingAgentEvaluator(config)

    test_cases = [
        {
            "instruction": "Add type hints to the function",
            "source_code": """\
def add(a, b):
    return a + b
""",
            "expected": """\
def add(a: int, b: int) -> int:
    return a + b
""",
            # Optional: pytest tests to run against the generated output
            "test_code": """\
def test_add_returns_int():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
""",
        },
        {
            "instruction": "Convert to async function",
            "source_code": """\
def get_user():
    return fetch_user()
""",
            "expected": """\
async def get_user():
    return await fetch_user()
""",
        },
        {
            "instruction": "Add a docstring",
            "source_code": """\
def multiply(a, b):
    return a * b
""",
            "expected": """\
def multiply(a, b):
    \"\"\"Multiply two numbers and return the result.\"\"\"
    return a * b
""",
            "test_code": """\
def test_multiply():
    assert multiply(3, 4) == 12

def test_multiply_zero():
    assert multiply(0, 99) == 0
""",
        },
    ]

    evaluator.evaluate(test_cases, save_results="eval_results.json")