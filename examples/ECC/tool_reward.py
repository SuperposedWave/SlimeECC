import ast
import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from examples.ECC import code_ops
from examples.ECC.ecc_reward import compute_minimum_distance
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward normalization: upper bound for min_distance via Singleton bound
# ---------------------------------------------------------------------------

def _singleton_bound(n: int, k: int) -> int:
    """Singleton bound: d <= n - k + 1."""
    return n - k + 1


def _normalize_score(min_distance: int, n: int, k: int) -> float:
    """Map min_distance into [0, 1] using the Singleton bound as ceiling."""
    upper = _singleton_bound(n, k)
    if upper <= 0:
        return 0.0
    return min(float(min_distance) / upper, 1.0)


def _target_closeness(pred: int | None, target: int | None) -> float:
    """A smooth [0, 1] closeness score for target-matching objectives."""
    if pred is None or target is None:
        return 0.0
    target = int(target)
    if target <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(int(pred) - target) / target)


# ---------------------------------------------------------------------------
# Historical sample registry for diversity penalty
# ---------------------------------------------------------------------------

_HISTORY_MAX_SIZE = int(os.getenv("ECC_HISTORY_MAX_SIZE", "16384"))
_DIVERSITY_PENALTY = float(os.getenv("ECC_DIVERSITY_PENALTY", "0.1"))

_history_lock = threading.Lock()
_history: OrderedDict[bytes, int] = OrderedDict()


def _matrix_key(G: np.ndarray) -> bytes:
    """Canonical key: RREF rows flattened to bytes for O(1) lookup."""
    from examples.ECC.code_ops import rref_gf2
    rref, _ = rref_gf2(G)
    return b"matrix:" + rref.tobytes()


def _response_key(response: str, steps: list["ParsedStep"] | None = None) -> bytes:
    """Canonical key for failed samples based on the normalized construction program."""
    if steps:
        normalized = "\n".join(step.source.strip() for step in steps if step.source.strip())
    else:
        normalized = _get_response_code(response)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return b"response:" + digest


def _check_and_register_key(key: bytes) -> bool:
    """Return True if this key is a duplicate; register it either way."""
    with _history_lock:
        if key in _history:
            _history[key] += 1
            _history.move_to_end(key)
            return True
        _history[key] = 1
        if len(_history) > _HISTORY_MAX_SIZE:
            _history.popitem(last=False)
        return False


def _check_and_register_matrix(G: np.ndarray) -> bool:
    return _check_and_register_key(_matrix_key(G))


def _check_and_register_response(response: str, steps: list["ParsedStep"] | None = None) -> bool:
    return _check_and_register_key(_response_key(response, steps))

# Fill this with the exact exported names from code_ops.__all__ when ready.
DEFAULT_ALLOWED_FUNCTIONS = {
    "augment_code",
    "build_family",
    "cyclic_code_binary",
    "code_complement",
    "concatenate_codes",
    "concatenated_code",
    "constacyclic_code_binary",
    "construction_x",
    "construction_x3",
    "construction_x3u",
    "construction_x_chain",
    "construction_xx",
    "construction_y1",
    "direct_product",
    "direct_sum",
    "direct_sum_many",
    "double_circulant_code_binary",
    "dual_code",
    "expurgate_code_list",
    "expurgate_code_odd_weight",
    "expurgate_weight_code",
    "extend_code",
    "extend_code_n",
    "from_generator_matrix",
    "from_parity_check_matrix_binary",
    "hamming_code_binary",
    "intersection_code",
    "juxtaposition",
    "lengthen_code",
    "pad_code",
    "plotkin_sum",
    "plotkin_sum_three",
    "product_code",
    "puncture_code",
    "puncture_code_set",
    "shorten_code",
    "shorten_code_set",
    "simplex_code_binary",
    "zinoviev_code",
}

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_VAR_RE = re.compile(r"^c\d+$")


@dataclass
class ParsedStep:
    target: str | None
    func_name: str
    args: list[Any]
    kwargs: dict[str, Any]
    source: str


def _metadata(sample: Sample) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _set_diagnostics(sample: Sample, **diagnostics: Any) -> None:
    metadata = _metadata(sample)
    metadata.update(diagnostics)
    sample.metadata = metadata


def _get_response_code(response: str) -> str:
    match = CODE_BLOCK_RE.search(response)
    return match.group(1).strip() if match else response.strip()


def _safe_literal_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        if CODE_VAR_RE.match(node.id):
            return node.id
        raise ValueError(f"Only code variables like c0, c1, ... are allowed, got {node.id}.")

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [_safe_literal_eval(elt) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal_eval(elt) for elt in node.elts)

    if isinstance(node, ast.Dict):
        return {
            _safe_literal_eval(key): _safe_literal_eval(value)
            for key, value in zip(node.keys, node.values, strict=False)
        }

    try:
        return ast.literal_eval(node)
    except Exception as exc:
        raise ValueError(f"Unsupported non-literal argument: {ast.unparse(node)}") from exc


def _parse_call_expr(call: ast.Call) -> tuple[str, list[Any], dict[str, Any]]:
    if not isinstance(call.func, ast.Name):
        raise ValueError("Only direct function calls like `foo(...)` are allowed.")

    func_name = call.func.id
    args = [_safe_literal_eval(arg) for arg in call.args]
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("Star-kwargs are not allowed.")
        kwargs[keyword.arg] = _safe_literal_eval(keyword.value)
    return func_name, args, kwargs


def _collect_code_vars(value: Any) -> set[str]:
    if isinstance(value, str) and CODE_VAR_RE.match(value):
        return {value}
    if isinstance(value, list):
        refs: set[str] = set()
        for item in value:
            refs |= _collect_code_vars(item)
        return refs
    if isinstance(value, tuple):
        refs: set[str] = set()
        for item in value:
            refs |= _collect_code_vars(item)
        return refs
    if isinstance(value, dict):
        refs: set[str] = set()
        for key, val in value.items():
            refs |= _collect_code_vars(key)
            refs |= _collect_code_vars(val)
        return refs
    return set()


def parse_tool_steps(response: str) -> list[ParsedStep]:
    code = _get_response_code(response)
    if not code:
        raise ValueError("Empty response.")

    tree = ast.parse(code)
    steps: list[ParsedStep] = []

    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module != "code_ops":
                raise ValueError("Only `from code_ops import ...` is allowed.")
            if any(alias.name == "*" for alias in stmt.names):
                raise ValueError("Wildcard imports are not allowed.")
            continue

        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple assignments like `c1 = foo(...)` are allowed.")
            if not CODE_VAR_RE.match(stmt.targets[0].id):
                raise ValueError("Assigned variables must be named c0, c1, c2, ...")
            if not isinstance(stmt.value, ast.Call):
                raise ValueError("Assignments must come from function calls.")
            func_name, args, kwargs = _parse_call_expr(stmt.value)
            steps.append(
                ParsedStep(
                    target=stmt.targets[0].id,
                    func_name=func_name,
                    args=args,
                    kwargs=kwargs,
                    source=ast.unparse(stmt),
                )
            )
            continue

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func_name, args, kwargs = _parse_call_expr(stmt.value)
            steps.append(
                ParsedStep(
                    target=None,
                    func_name=func_name,
                    args=args,
                    kwargs=kwargs,
                    source=ast.unparse(stmt),
                )
            )
            continue

        raise ValueError(
            "Only import lines, plain function calls, and simple assignments from function calls are allowed."
        )

    if not steps:
        raise ValueError("No executable tool steps found.")
    return steps


def validate_tool_steps(steps: list[ParsedStep], allowed_functions: set[str]) -> tuple[bool, str | None]:
    known_vars: set[str] = set()

    for step in steps:
        if step.func_name not in allowed_functions:
            return False, f"forbidden_function:{step.func_name}"

        for arg in step.args:
            if isinstance(arg, str) and CODE_VAR_RE.match(arg) and arg not in known_vars:
                return False, f"unknown_variable:{arg}"
        for value in step.kwargs.values():
            if isinstance(value, str) and CODE_VAR_RE.match(value) and value not in known_vars:
                return False, f"unknown_variable:{value}"

        if step.target is not None:
            known_vars.add(step.target)

    return True, None


def resolve_allowed_functions(sample: Sample) -> set[str]:
    metadata = _metadata(sample)
    value = metadata.get("allowed_functions")
    if value is None:
        return set(DEFAULT_ALLOWED_FUNCTIONS)
    if isinstance(value, list):
        return {str(item) for item in value}
    raise ValueError("metadata['allowed_functions'] must be a list of function names.")


def analyze_tool_step_connectivity(steps: list[ParsedStep]) -> dict[str, Any]:
    assigned_targets = [step.target for step in steps if step.target is not None]
    if len(assigned_targets) <= 1:
        return {
            "ok": True,
            "final_target": assigned_targets[-1] if assigned_targets else None,
            "unused_assigned_vars": [],
            "final_is_standalone": False,
        }

    deps_by_target: dict[str, set[str]] = {}
    for step in steps:
        if step.target is None:
            continue
        refs = set()
        for arg in step.args:
            refs |= _collect_code_vars(arg)
        for value in step.kwargs.values():
            refs |= _collect_code_vars(value)
        deps_by_target[step.target] = refs

    final_target = assigned_targets[-1]
    reachable: set[str] = set()
    stack = [final_target]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(sorted(deps_by_target.get(current, set()) - reachable))

    unused_assigned_vars = [target for target in assigned_targets[:-1] if target not in reachable]
    final_is_standalone = len(deps_by_target.get(final_target, set())) == 0

    return {
        "ok": not final_is_standalone and not unused_assigned_vars,
        "final_target": final_target,
        "unused_assigned_vars": unused_assigned_vars,
        "final_is_standalone": final_is_standalone,
    }


def _resolve_runtime_value(value: Any, env: dict[str, Any]) -> Any:
    if isinstance(value, str) and CODE_VAR_RE.match(value):
        if value not in env:
            raise ValueError(f"Unknown variable {value}.")
        return env[value]
    if isinstance(value, list):
        return [_resolve_runtime_value(item, env) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_runtime_value(item, env) for item in value)
    if isinstance(value, dict):
        return {
            _resolve_runtime_value(key, env): _resolve_runtime_value(val, env)
            for key, val in value.items()
        }
    return value


def execute_tool_steps(steps: list[ParsedStep], sample: Sample) -> dict[str, Any]:
    dispatch = {name: getattr(code_ops, name) for name in DEFAULT_ALLOWED_FUNCTIONS if hasattr(code_ops, name)}
    env: dict[str, Any] = {}
    last_target: str | None = None

    metadata = _metadata(sample)
    expected_n = metadata.get("n")
    expected_k = metadata.get("k")

    try:
        for step in steps:
            if step.func_name not in dispatch:
                return {
                    "ok": False,
                    "category": f"missing_dispatch:{step.func_name}",
                    "message": f"Function {step.func_name} is not available in code_ops.",
                }

            resolved_args = [_resolve_runtime_value(arg, env) for arg in step.args]
            resolved_kwargs = {key: _resolve_runtime_value(val, env) for key, val in step.kwargs.items()}
            result = dispatch[step.func_name](*resolved_args, **resolved_kwargs)

            if step.target is not None:
                env[step.target] = result
                last_target = step.target

        if last_target is None:
            return {
                "ok": False,
                "category": "no_final_code",
                "message": "No assigned code variable was produced.",
            }

        final_code = env[last_target]
        if not isinstance(final_code, code_ops.LinearCode):
            return {
                "ok": False,
                "category": "final_not_linear_code",
                "message": f"Final object {last_target} is not a LinearCode.",
            }

        min_distance = compute_minimum_distance(final_code.G.tolist())
        n_match = expected_n is None or final_code.n == int(expected_n)
        k_match = expected_k is None or final_code.k == int(expected_k)

        if n_match and k_match:
            category = "success"
            message = None
        elif not n_match and not k_match:
            category = "wrong_length_and_dimension"
            message = f"Expected (n, k)=({expected_n}, {expected_k}), got ({final_code.n}, {final_code.k})."
        elif not n_match:
            category = "wrong_length"
            message = f"Expected n={expected_n}, got n={final_code.n}."
        else:
            category = "wrong_dimension"
            message = f"Expected k={expected_k}, got k={final_code.k}."

        return {
            "ok": True,
            "category": category,
            "message": message,
            "final_code": final_code,
            "n": final_code.n,
            "k": final_code.k,
            "min_distance": min_distance,
            "n_match": n_match,
            "k_match": k_match,
        }
    except Exception as exc:
        return {
            "ok": False,
            "category": "execution_failed",
            "message": str(exc),
        }


def _resolve_targets(sample: Sample) -> dict[str, Any]:
    metadata = _metadata(sample)
    return {
        "target_distance": metadata.get("target_min_distance", metadata.get("d")),
        "target_n": metadata.get("n"),
        "target_k": metadata.get("k"),
    }


def _format_reward(
    *,
    score: float,
    category: str,
    parsed_steps: int = 0,
    min_distance: int | None = None,
    target_distance: int | None = None,
    n: int | None = None,
    k: int | None = None,
    raw_score: float | None = None,
    duplicate: bool = False,
    message: str | None = None,
    syntax_score: float | None = None,
    executable_score: float | None = None,
    length_score: float | None = None,
    dimension_score: float | None = None,
    distance_score: float | None = None,
) -> dict[str, Any]:
    reward = {
        "score": float(score),
        "category": category,
        "parsed_steps": parsed_steps,
        "min_distance": min_distance,
        "target_distance": target_distance,
        "n": n,
        "k": k,
        "duplicate": duplicate,
    }
    if raw_score is not None:
        reward["raw_score"] = float(raw_score)
    if message is not None:
        reward["message"] = message
    if syntax_score is not None:
        reward["syntax_score"] = float(syntax_score)
    if executable_score is not None:
        reward["executable_score"] = float(executable_score)
    if length_score is not None:
        reward["length_score"] = float(length_score)
    if dimension_score is not None:
        reward["dimension_score"] = float(dimension_score)
    if distance_score is not None:
        reward["distance_score"] = float(distance_score)
    return reward


async def custom_rm(args, sample: Sample | list[Sample], **kwargs):
    if isinstance(sample, list):
        return [await custom_rm(args, item, **kwargs) for item in sample]

    diversity_penalty = float(os.getenv("ECC_DIVERSITY_PENALTY", str(_DIVERSITY_PENALTY)))

    targets = _resolve_targets(sample)
    try:
        allowed_functions = resolve_allowed_functions(sample)
        steps = parse_tool_steps(sample.response)
        valid, reason = validate_tool_steps(steps, allowed_functions)
        if not valid:
            duplicate = _check_and_register_response(sample.response, steps)
            score = -0.3 - (diversity_penalty if duplicate else 0.0)
            category = f"{reason or 'validation_error'}_duplicate" if duplicate else reason or "validation_error"
            reward = _format_reward(
                score=score,
                category=category,
                parsed_steps=len(steps),
                duplicate=duplicate,
                message="Function whitelist validation failed.",
            )
            _set_diagnostics(sample, rollout_correct=False, ecc_tool_reward=reward, ecc_steps=[s.source for s in steps])
            return reward

        connectivity = analyze_tool_step_connectivity(steps)
        if not connectivity["ok"]:
            if connectivity["final_is_standalone"]:
                category = "disconnected_final_step"
                message = "Final assigned code does not depend on any earlier code variable."
            else:
                category = "unused_prefix_steps"
                message = (
                    "Some earlier assigned code variables are disconnected from the final result: "
                    + ", ".join(connectivity["unused_assigned_vars"])
                )
            duplicate = _check_and_register_response(sample.response, steps)
            score = -0.25 - (diversity_penalty if duplicate else 0.0)
            category = f"{category}_duplicate" if duplicate else category
            reward = _format_reward(
                score=score,
                category=category,
                parsed_steps=len(steps),
                duplicate=duplicate,
                message=message,
            )
            _set_diagnostics(
                sample,
                rollout_correct=False,
                ecc_tool_reward=reward,
                ecc_steps=[s.source for s in steps],
                ecc_unused_assigned_vars=connectivity["unused_assigned_vars"],
                ecc_final_target=connectivity["final_target"],
            )
            return reward

        try:
            execution = execute_tool_steps(steps, sample)
        except NotImplementedError as exc:
            duplicate = _check_and_register_response(sample.response, steps)
            score = -0.1 - (diversity_penalty if duplicate else 0.0)
            reward = _format_reward(
                score=score,
                category="parsed_but_not_executed_duplicate" if duplicate else "parsed_but_not_executed",
                parsed_steps=len(steps),
                duplicate=duplicate,
                message=str(exc),
            )
            _set_diagnostics(sample, rollout_correct=False, ecc_tool_reward=reward, ecc_steps=[s.source for s in steps])
            return reward

        if not execution.get("ok", False):
            duplicate = _check_and_register_response(sample.response, steps)
            base_category = str(execution.get("category", "execution_failed"))
            score = -0.2 - (diversity_penalty if duplicate else 0.0)
            reward = _format_reward(
                score=score,
                category=f"{base_category}_duplicate" if duplicate else base_category,
                parsed_steps=len(steps),
                duplicate=duplicate,
                message=str(execution.get("message", "Execution failed.")),
            )
            _set_diagnostics(sample, rollout_correct=False, ecc_tool_reward=reward, ecc_steps=[s.source for s in steps])
            return reward

        min_distance = execution.get("min_distance")
        code_n = execution.get("n")
        code_k = execution.get("k")
        target_distance = execution.get("target_distance", targets["target_distance"])
        execution_category = str(execution.get("category", "success"))

        raw_score = float(min_distance or 0.0)
        syntax_score = 1.0
        executable_score = 1.0
        length_score = _target_closeness(code_n, targets["target_n"])
        dimension_score = _target_closeness(code_k, targets["target_k"])
        nk_exact_match = bool(execution.get("n_match", False) and execution.get("k_match", False))
        distance_score = _normalize_score(min_distance or 0, code_n or 1, code_k or 0) if nk_exact_match else 0.0

        score = (
            0.25 * syntax_score
            + 0.25 * executable_score
            + 0.25 * length_score
            + 0.25 * dimension_score
            + 0.5 * distance_score
        )

        # Ensure any dimension mismatch gets a strictly negative reward.
        if not bool(execution.get("k_match", False)):
            score = -max(float(score), 0.05)

        final_code = execution.get("final_code")
        duplicate = False
        if final_code is not None and isinstance(final_code, code_ops.LinearCode):
            duplicate = _check_and_register_matrix(final_code.G)
            if duplicate:
                score = max(score - diversity_penalty, 0.0) if score >= 0.0 else score - diversity_penalty

        category = f"{execution_category}_duplicate" if duplicate else execution_category
        rollout_correct = score > 0.0

        reward = _format_reward(
            score=score,
            raw_score=raw_score,
            category=category,
            parsed_steps=len(steps),
            min_distance=min_distance,
            target_distance=target_distance,
            n=code_n,
            k=code_k,
            duplicate=duplicate,
            message=execution.get("message"),
            syntax_score=syntax_score,
            executable_score=executable_score,
            length_score=length_score,
            dimension_score=dimension_score,
            distance_score=distance_score,
        )
        _set_diagnostics(
            sample,
            rollout_correct=rollout_correct,
            ecc_tool_reward=reward,
            ecc_steps=[s.source for s in steps],
        )
        return reward
    except Exception as exc:
        logger.info("ECC tool reward failed: %s", exc)
        duplicate = _check_and_register_response(sample.response)
        score = -0.3 - (diversity_penalty if duplicate else 0.0)
        reward = _format_reward(
            score=score,
            category="parse_error_duplicate" if duplicate else "parse_error",
            parsed_steps=0,
            duplicate=duplicate,
            message=str(exc),
        )
        _set_diagnostics(sample, rollout_correct=False, ecc_tool_reward=reward, ecc_error=str(exc))
        return reward
