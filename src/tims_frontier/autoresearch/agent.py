from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pprint
import re
import socket
import time
from typing import Any, Mapping
import urllib.error
import urllib.request

from .archive import AutoResearchArchive
from .common import REPO_ROOT, read_json_if_exists, write_json
from .config import load_autoresearch_config
from .loop import init_run, load_current_proposal, run_step
from .models import Proposal, SUPPORTED_EDIT_TYPES
from .summary import build_next_context_summary
from .validator import ProposalValidationError, infer_allowed_edit_types, validate_proposal


class LLMBackendError(RuntimeError):
    """Raised when the configured LLM backend fails or returns unusable output."""


EDIT_TYPE_ALIASES = {
    "parameter_range_narrowing": "scalar_tune",
    "parameter_range_widening": "scalar_tune",
    "parameter_range_shift": "scalar_tune",
    "parameter_tune": "scalar_tune",
    "numeric_tune": "scalar_tune",
    "structure_expansion": "structural_expand",
    "structure_reduction": "structural_reduce",
    "coupling_rule_change": "coupling_change",
    "correlation_change": "coupling_change",
    "distribution_swap": "distribution_change",
    "distribution_form_change": "distribution_change",
}


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    use_json_mode: bool
    max_retries: int
    retry_backoff_seconds: float
    retry_max_backoff_seconds: float

    def redacted(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "use_json_mode": self.use_json_mode,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "retry_max_backoff_seconds": self.retry_max_backoff_seconds,
        }


def _coerce_env_names(raw_value: Any, *, fallback: list[str]) -> list[str]:
    if raw_value is None:
        return list(fallback)
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    return list(fallback)


def _load_env_file(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    env_path = Path(path)
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        resolved_value = value.strip().strip("'").strip('"')
        loaded[key.strip()] = resolved_value
    return loaded


def _resolve_env(env_names: list[str], env: Mapping[str, str], *, default: str | None = None) -> str | None:
    for env_name in env_names:
        value = env.get(env_name)
        if value:
            return value
    return default


def resolve_llm_settings(runtime_config: Mapping[str, Any], *, env: Mapping[str, str] | None = None) -> LLMSettings:
    llm_config = dict(runtime_config.get("llm", {}))
    environment = dict(os.environ if env is None else env)
    env_file_value = llm_config.get("env_file")
    env_file_path = None
    if env_file_value:
        env_file_path = Path(str(env_file_value))
        if not env_file_path.is_absolute():
            env_file_path = (REPO_ROOT / env_file_path).resolve()
    file_environment = _load_env_file(env_file_path)
    merged_environment = dict(file_environment)
    merged_environment.update(environment)

    api_key = _resolve_env(
        _coerce_env_names(
            llm_config.get("api_key_env"),
            fallback=["AUTORESEARCH_LLM_API_KEY", "MOONSHOT_API_KEY"],
        ),
        merged_environment,
        default=llm_config.get("api_key"),
    )
    if not api_key:
        raise LLMBackendError(
            "No LLM API key found. Set AUTORESEARCH_LLM_API_KEY or MOONSHOT_API_KEY before running ai-step."
        )

    base_url = _resolve_env(
        _coerce_env_names(
            llm_config.get("base_url_env"),
            fallback=["AUTORESEARCH_LLM_BASE_URL", "MOONSHOT_BASE_URL"],
        ),
        merged_environment,
        default=str(llm_config.get("default_base_url", "https://api.moonshot.ai/v1")),
    )
    model = _resolve_env(
        _coerce_env_names(
            llm_config.get("model_env"),
            fallback=["AUTORESEARCH_LLM_MODEL"],
        ),
        merged_environment,
        default=str(llm_config.get("default_model", "kimi-k2.5")),
    )
    return LLMSettings(
        base_url=str(base_url).rstrip("/"),
        api_key=str(api_key),
        model=str(model),
        temperature=float(llm_config.get("temperature", 1.0)),
        max_tokens=int(llm_config.get("max_tokens", 16000)),
        timeout_seconds=float(llm_config.get("timeout_seconds", 120.0)),
        use_json_mode=bool(llm_config.get("use_json_mode", False)),
        max_retries=max(0, int(llm_config.get("max_retries", 3))),
        retry_backoff_seconds=max(0.0, float(llm_config.get("retry_backoff_seconds", 5.0))),
        retry_max_backoff_seconds=max(0.0, float(llm_config.get("retry_max_backoff_seconds", 30.0))),
    )


def _normalize_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("text"):
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _decode_json_like(text: str) -> Mapping[str, Any]:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
        unfenced = _strip_code_fence(stripped)
        if unfenced != stripped:
            candidates.append(unfenced)

    decoder = json.JSONDecoder()
    brace_positions = [match.start() for match in re.finditer(r"\{", stripped)]
    for index in brace_positions:
        fragment = stripped[index:]
        try:
            parsed, end_index = decoder.raw_decode(fragment)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(fragment[:end_index])

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                loaded = ast.literal_eval(candidate)
            except Exception:
                continue
        if not isinstance(loaded, Mapping):
            continue
        payload = loaded.get("proposal") if isinstance(loaded.get("proposal"), Mapping) else loaded
        if isinstance(payload, Mapping):
            return dict(payload)
    raise LLMBackendError("Unable to parse a JSON proposal object from the LLM response.")


def request_proposal_from_llm(
    messages: list[dict[str, str]],
    settings: LLMSettings,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if settings.use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        f"{settings.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    raw_body = ""
    total_attempts = settings.max_retries + 1
    for attempt_index in range(total_attempts):
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            is_retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if not is_retryable or attempt_index >= settings.max_retries:
                raise LLMBackendError(f"LLM backend returned HTTP {exc.code}: {error_body}") from exc
            _sleep_before_retry(settings, attempt_index)
        except (TimeoutError, socket.timeout) as exc:
            if attempt_index >= settings.max_retries:
                raise LLMBackendError(
                    f"LLM backend timed out after {total_attempts} attempts at {settings.timeout_seconds:.1f}s each."
                ) from exc
            _sleep_before_retry(settings, attempt_index)
        except urllib.error.URLError as exc:
            reason_text = str(getattr(exc, "reason", exc)).lower()
            is_retryable = any(token in reason_text for token in ["timed out", "timeout", "tempor", "reset", "unreachable"])
            if not is_retryable or attempt_index >= settings.max_retries:
                raise LLMBackendError(f"Unable to reach the configured LLM backend: {exc}") from exc
            _sleep_before_retry(settings, attempt_index)
    else:
        raise LLMBackendError("LLM backend request failed before a response body was received.")

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise LLMBackendError(f"LLM backend returned non-JSON content: {raw_body[:500]}") from exc

    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMBackendError("LLM backend response did not contain any choices.")
    message = dict(choices[0].get("message") or {})
    content = _normalize_message_text(message.get("content"))
    reasoning_content = _normalize_message_text(message.get("reasoning_content"))
    proposal_payload = _decode_json_like(content)
    return {
        "model": parsed.get("model", settings.model),
        "content": content,
        "reasoning_content": reasoning_content,
        "proposal": proposal_payload,
        "raw_response": parsed,
    }


def _sleep_before_retry(settings: LLMSettings, attempt_index: int) -> None:
    backoff = settings.retry_backoff_seconds * (2 ** attempt_index)
    capped_backoff = min(backoff, settings.retry_max_backoff_seconds)
    if capped_backoff > 0:
        time.sleep(capped_backoff)


def _normalize_proposal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(payload)))
    raw_edit_type = normalized.get("edit_type")
    if isinstance(raw_edit_type, str):
        edit_key = raw_edit_type.strip()
        normalized["edit_type"] = EDIT_TYPE_ALIASES.get(edit_key, edit_key)
    return normalized


def _load_parent_proposal_from_state(state: Mapping[str, Any]) -> Proposal | None:
    payload_path = state.get("last_attempted_proposal_path")
    if not payload_path:
        return None
    payload = read_json_if_exists(str(payload_path))
    if payload is None:
        return None
    return Proposal.from_mapping(payload)


def _prevalidate_agent_proposal(
    proposal_payload: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    state: Mapping[str, Any],
) -> Proposal:
    normalized_payload = dict(proposal_payload)
    proposal = Proposal.from_mapping(normalized_payload)
    parent_proposal = _load_parent_proposal_from_state(state)
    try:
        validate_proposal(proposal, runtime_config, parent_proposal=parent_proposal)
        return proposal
    except ProposalValidationError as exc:
        allowed_types, diagnostics = infer_allowed_edit_types(
            proposal,
            runtime_config,
            parent_proposal=parent_proposal,
        )
        if proposal.edit_type not in allowed_types and len(allowed_types) == 1:
            corrected_payload = dict(normalized_payload)
            corrected_payload["edit_type"] = next(iter(allowed_types))
            corrected_proposal = Proposal.from_mapping(corrected_payload)
            validate_proposal(corrected_proposal, runtime_config, parent_proposal=parent_proposal)
            return corrected_proposal
        raise ProposalValidationError(f"{exc}. Inferred allowed edit types: {sorted(allowed_types)}. Diagnostics: {diagnostics}") from exc


def _latest_round_payload(state: Mapping[str, Any], filename: str) -> dict[str, Any] | None:
    for round_record in reversed(list(state.get("history", []))):
        round_dir = round_record.get("round_dir")
        if not round_dir:
            continue
        payload = read_json_if_exists(Path(str(round_dir)) / filename)
        if payload is not None:
            return payload
    return None


def build_agent_context(
    runtime_config: Mapping[str, Any],
    archive: AutoResearchArchive,
    state: Mapping[str, Any],
    *,
    train_path: str | Path,
    program_path: str | Path,
) -> dict[str, Any]:
    train_source = Path(train_path).read_text(encoding="utf-8")
    current_train_proposal = load_current_proposal(train_path).to_dict()
    last_attempted_payload = None
    if state.get("last_attempted_proposal_path"):
        last_attempted_payload = read_json_if_exists(str(state["last_attempted_proposal_path"]))

    last_round_record = list(state.get("history", []))[-1] if state.get("history") else None
    next_context = (
        build_next_context_summary(state, last_round_record, runtime_config) if last_round_record is not None else None
    )
    crash_context = _latest_round_payload(state, "crash_context.json")
    failure_payload = _latest_round_payload(state, "failure.json")
    baseline_payload = read_json_if_exists(archive.baseline_path)
    program_text = Path(program_path).read_text(encoding="utf-8")
    return {
        "program_md": program_text,
        "train_source": train_source,
        "current_train_proposal": current_train_proposal,
        "last_attempted_proposal": last_attempted_payload,
        "next_context_summary": next_context,
        "baseline_payload": baseline_payload,
        "latest_crash_context": crash_context,
        "latest_failure_payload": failure_payload,
        "run_state_excerpt": {
            "run_id": state.get("run_id"),
            "branch_name": state.get("branch_name"),
            "round_index": state.get("round_index"),
            "best_commit": state.get("best_commit"),
            "best_score": state.get("best_score"),
            "best_proposal_id": state.get("best_proposal_id"),
            "baseline_round_id": state.get("baseline_round_id"),
            "baseline_hv": state.get("baseline_hv"),
            "last_attempted_proposal_id": state.get("last_attempted_proposal_id"),
        },
    }


def build_agent_messages(context: Mapping[str, Any]) -> list[dict[str, str]]:
    allowed_edit_types = ", ".join(sorted(SUPPORTED_EDIT_TYPES))
    system_prompt = (
        "You are the single-file autoresearch agent for heterogeneous nanodot reservoir family search.\n"
        "You must follow program.md exactly.\n"
        "You are only proposing the next CURRENT_PROPOSAL for train.py.\n"
        "Return JSON only, with no markdown fences and no extra prose.\n"
        "Return either the raw proposal object or {'proposal': <proposal>}.\n"
        "The proposal must include proposal_id, parent_proposal_id, edit_type, primary_edit, rationale, "
        "expected_effect, family_definition, sampling_plan, notes, and metadata if needed.\n"
        f"edit_type MUST be exactly one of: {allowed_edit_types}.\n"
        "Do not invent new edit_type labels like parameter_range_narrowing.\n"
        "Every new round must make exactly one minimal semantic edit relative to the last attempted proposal.\n"
        "If the previous round crashed, repair only the smallest issue needed and use the traceback tail.\n"
        "Do not propose evaluator/runtime/framework changes. Do not mention editing any file other than train.py."
    )

    user_sections = [
        "program.md:\n" + str(context["program_md"]),
        "Current train.py source:\n```python\n" + str(context["train_source"]) + "\n```",
        "Current CURRENT_PROPOSAL on disk:\n" + json.dumps(context["current_train_proposal"], ensure_ascii=False, indent=2),
        "Last attempted proposal (semantic parent for the next minimal edit):\n"
        + json.dumps(context.get("last_attempted_proposal"), ensure_ascii=False, indent=2),
        "Run state excerpt:\n" + json.dumps(context["run_state_excerpt"], ensure_ascii=False, indent=2),
        "Next-round context summary:\n" + json.dumps(context.get("next_context_summary"), ensure_ascii=False, indent=2),
        "Baseline snapshot:\n" + json.dumps(context.get("baseline_payload"), ensure_ascii=False, indent=2),
        "Latest crash context:\n" + json.dumps(context.get("latest_crash_context"), ensure_ascii=False, indent=2),
        "Latest failure payload:\n" + json.dumps(context.get("latest_failure_payload"), ensure_ascii=False, indent=2),
        (
            "Return contract:\n"
            "- Output valid JSON only.\n"
            "- The proposal must be complete, not a partial patch.\n"
            "- Keep the sampling plan stable unless changing it is the single primary semantic edit.\n"
            "- parent_proposal_id should normally equal the last_attempted_proposal proposal_id.\n"
            "- Keep all unsupported fields out of the response."
        ),
    ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]


def replace_current_proposal_in_train(train_path: str | Path, proposal_payload: Mapping[str, Any]) -> bool:
    path = Path(train_path)
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    assignment = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CURRENT_PROPOSAL":
                assignment = node
                break
        if assignment is not None:
            break
    if assignment is None or getattr(assignment, "end_lineno", None) is None:
        raise LLMBackendError("Unable to locate CURRENT_PROPOSAL in train.py for automatic rewrite.")

    formatted_payload = pprint.pformat(dict(proposal_payload), width=100, sort_dicts=False)
    replacement = f"CURRENT_PROPOSAL = {formatted_payload}\n"
    lines = source.splitlines(keepends=True)
    updated = "".join(lines[: assignment.lineno - 1] + [replacement] + lines[assignment.end_lineno :])
    if updated == source:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def ai_step(config_path: str | None = None) -> dict[str, Any]:
    runtime_config = load_autoresearch_config(config_path)
    try:
        archive, state = AutoResearchArchive.from_active(runtime_config)
    except FileNotFoundError:
        init_run(config_path)
        archive, state = AutoResearchArchive.from_active(runtime_config)

    is_bootstrap_round = (
        int(state.get("round_index", 0)) == 0
        and not state.get("history")
        and state.get("baseline_round_id") is None
        and state.get("last_attempted_proposal_id") is None
    )
    if is_bootstrap_round:
        bootstrap_proposal = load_current_proposal(REPO_ROOT / "train.py")
        if bootstrap_proposal.edit_type != "initial_seed":
            raise LLMBackendError(
                "The bootstrap ai-step requires train.py to contain an initial_seed proposal before baseline exists."
            )
        return run_step(config_path)

    settings = resolve_llm_settings(runtime_config)
    round_index = int(state["round_index"]) + 1
    round_dir = archive.round_dir(round_index)
    context = build_agent_context(
        runtime_config,
        archive,
        state,
        train_path=REPO_ROOT / "train.py",
        program_path=REPO_ROOT / "program.md",
    )
    messages = build_agent_messages(context)
    archive.write_round_json(round_index, "agent_context.json", context)
    archive.write_round_json(
        round_index,
        "agent_request.json",
        {
            "llm_settings": settings.redacted(),
            "messages": messages,
        },
    )

    try:
        response = request_proposal_from_llm(messages, settings)
    except Exception as exc:
        write_json(
            round_dir / "agent_failure.json",
            {"error_type": type(exc).__name__, "message": str(exc)},
        )
        raise

    normalized_proposal = _normalize_proposal_payload(response["proposal"])
    archive.write_round_json(
        round_index,
        "agent_response.json",
        {
            "model": response["model"],
            "reasoning_content": response["reasoning_content"],
            "content": response["content"],
            "proposal": normalized_proposal,
            "raw_response": response["raw_response"],
        },
    )
    try:
        validated_proposal = _prevalidate_agent_proposal(normalized_proposal, runtime_config, state)
    except (ProposalValidationError, ValueError) as exc:
        archive.write_round_json(
            round_index,
            "agent_validation_error.json",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "proposal": normalized_proposal,
            },
        )
        raise LLMBackendError(f"LLM returned an invalid proposal before execution: {exc}") from exc

    changed = replace_current_proposal_in_train(REPO_ROOT / "train.py", validated_proposal.to_dict())
    if not changed:
        raise LLMBackendError("The LLM returned a proposal identical to the current train.py payload.")
    archive.write_round_json(round_index, "agent_proposal.json", validated_proposal.to_dict())
    return run_step(config_path)


def ai_loop(config_path: str | None = None, *, iterations: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for _ in range(iterations):
        results.append(ai_step(config_path))
    return results
