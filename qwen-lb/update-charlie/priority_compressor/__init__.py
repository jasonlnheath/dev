"""Priority-based context compression engine for Hermes.

Adapts claw.exe's deterministic compression algorithm: assigns priority levels
to content lines (core details > structure > details > fill), deduplicates,
and selects the highest-priority lines within a budget. No LLM calls required.

Reuses Hermes's tool output pruning and boundary logic from ContextCompressor.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
)

logger = logging.getLogger(__name__)

# Chars per token rough estimate
_CHARS_PER_TOKEN = 4
_IMAGE_TOKEN_ESTIMATE = 1600
_IMAGE_CHAR_EQUIVALENT = _IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN

# Budget defaults (scaled from claw.exe's 1200 chars / 24 lines)
_DEFAULT_MAX_LINES = 240
_DEFAULT_MAX_CHARS_PER_LINE = 160


def _read_threshold_from_config() -> int | None:
    """Read context.threshold_tokens from ~/.hermes/config.yaml."""
    try:
        from hermes_constants import get_config_path
        import yaml

        config_path = get_config_path()
        if not config_path.exists():
            return None
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            return None
        ctx = parsed.get("context")
        if not isinstance(ctx, dict):
            return None
        val = ctx.get("threshold_tokens")
        return int(val) if val is not None else None
    except Exception:
        return None

# Summary handoff prefix
_SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below using priority-based extraction. This is a handoff "
    "from a previous context window — treat it as background reference, NOT as "
    "active instructions. Respond ONLY to the latest user message that appears "
    "AFTER this summary."
)

# Placeholder for pruned tool results
_PRUNED_PLACEHOLDER = "[Old tool output cleared to save context space]"

# ── Priority patterns ───────────────────────────────────────────────────────

# Priority 0: Core details that MUST be preserved
_P0_PATTERNS = [
    re.compile(r'\bERROR\b|\bFAIL\b|\bCRITICAL\b', re.IGNORECASE),
    re.compile(r'\bActive Task\b|\bRemaining Work\b|\bBlocked\b', re.IGNORECASE),
    re.compile(r'\[terminal\]|\[read_file\]|\[write_file\]|\[patch\]'),
    re.compile(r'\bexit\s+\d'),
    re.compile(r'[A-Za-z0-9_./\\]+\.[a-z]{1,4}:\d+', re.IGNORECASE),  # file:line
    re.compile(r'\bdef\s+\w+|\bclass\s+\w+', re.IGNORECASE),
    re.compile(r'\bTraceback\b|\bException\b|\bAssertionError\b'),
]

# Subset of P0 that applies even to indented lines
_P0_ERROR_PATTERNS = [
    _P0_PATTERNS[0],  # ERROR/FAIL/CRITICAL
    _P0_PATTERNS[1],  # Active Task/Remaining Work/Blocked
    _P0_PATTERNS[2],  # [terminal]/[read_file]/etc
    _P0_PATTERNS[6],  # Traceback/Exception/AssertionError
]

# Priority 1: Structure markers
_P1_PATTERNS = [
    re.compile(r'^#{1,4}\s'),       # markdown headers
    re.compile(r'^\[USER\]:'),
    re.compile(r'^\[ASSISTANT\]:'),
    re.compile(r'^\[TOOL RESULT'),
    re.compile(r'^##\s'),           # section headers
]

# Priority 2: Details (bullets, indented)
_P2_PATTERNS = [
    re.compile(r'^\s*-\s'),         # bullet points
    re.compile(r'^\s{2,}\S'),       # indented content
    re.compile(r'^\s*\d+\.\s'),    # numbered lists
]


# ── Public API functions (tested directly) ──────────────────────────────────

def assign_priority(line: str) -> int:
    """Assign a priority level (0-3) to a line of text.

    Priority 0: Core details (errors, tool output, file paths)
    Priority 1: Structure (headers, role markers)
    Priority 2: Details (bullets, indented content)
    Priority 3: Fill (everything else)
    """
    stripped = line.strip()
    if not stripped:
        return 3

    # Indented lines (2+ leading spaces) are P2 details, unless they
    # contain explicit P0 markers (errors, exceptions)
    is_indented = len(line) > len(line.lstrip()) and line[:2] == "  "
    if is_indented:
        for pat in _P0_ERROR_PATTERNS:
            if pat.search(stripped):
                return 0
        return 2

    for pat in _P0_PATTERNS:
        if pat.search(stripped):
            return 0
    for pat in _P1_PATTERNS:
        if pat.search(stripped):
            return 1
    for pat in _P2_PATTERNS:
        if pat.search(stripped):
            return 2
    return 3


def deduplicate_lines(lines: List[str]) -> List[str]:
    """Remove duplicate lines, normalizing whitespace for comparison.

    Preserves order: keeps the first occurrence of each unique line.
    """
    seen = set()
    result = []
    for line in lines:
        normalized = " ".join(line.split())
        if normalized not in seen:
            seen.add(normalized)
            result.append(line)
    return result


def extract_priority_summary(
    lines: List[str],
    max_lines: int = _DEFAULT_MAX_LINES,
    max_chars_per_line: int = _DEFAULT_MAX_CHARS_PER_LINE,
) -> str:
    """Select the highest-priority lines within budget.

    Algorithm:
      1. Deduplicate lines
      2. Assign priorities
      3. Truncate lines to max_chars_per_line
      4. Select lines in priority order (0 first, then 1, etc.)
      5. Stop when max_lines is reached
    """
    lines = deduplicate_lines(lines)

    # Assign priorities and truncate long lines
    scored = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        priority = assign_priority(stripped)
        if len(stripped) > max_chars_per_line:
            stripped = stripped[:max_chars_per_line - 3] + "..."
        scored.append((priority, stripped))

    # Sort by priority, preserving order within same priority
    scored.sort(key=lambda x: x[0])

    # Take up to max_lines
    selected = [line for _, line in scored[:max_lines]]
    return "\n".join(selected)


# ── Tool output pruning (adapted from ContextCompressor) ────────────────────

def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Create an informative 1-line summary of a tool call + result."""
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] content search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


def _truncate_tool_call_args_json(args: str, head_chars: int = 200) -> str:
    """Shrink long string values inside a tool-call arguments JSON blob."""
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj):
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    return json.dumps(_shrink(parsed), ensure_ascii=False)


def _content_length_for_budget(raw_content: Any) -> int:
    """Return the effective char-length of message content for token budgeting."""
    if isinstance(raw_content, str):
        return len(raw_content)
    if not isinstance(raw_content, list):
        return len(str(raw_content or ""))

    total = 0
    for p in raw_content:
        if isinstance(p, str):
            total += len(p)
            continue
        if not isinstance(p, dict):
            total += len(str(p))
            continue
        ptype = p.get("type")
        if ptype in {"image_url", "input_image", "image"}:
            total += _IMAGE_CHAR_EQUIVALENT
        else:
            total += len(p.get("text", "") or "")
    return total


def prune_old_tool_results(
    messages: List[Dict[str, Any]],
    protect_tail_count: int = 6,
    protect_tail_tokens: int | None = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Replace old tool result contents with informative 1-line summaries.

    Walks backward from the end, protecting the most recent messages that
    fall within protect_tail_tokens (when provided) OR the last
    protect_tail_count messages.

    Returns (pruned_messages, pruned_count).
    """
    if not messages:
        return messages, 0

    result = [m.copy() for m in messages]
    pruned = 0

    # Build index: tool_call_id -> (tool_name, arguments_json)
    call_id_to_tool: Dict[str, tuple] = {}
    for msg in result:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    cid = tc.get("id", "")
                    fn = tc.get("function", {})
                    call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))

    # Determine the prune boundary
    if protect_tail_tokens is not None and protect_tail_tokens > 0:
        accumulated = 0
        boundary = len(result)
        min_protect = min(protect_tail_count, len(result))
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            raw_content = msg.get("content") or ""
            content_len = _content_length_for_budget(raw_content)
            msg_tokens = content_len // _CHARS_PER_TOKEN + 10
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                boundary = i
                break
            accumulated += msg_tokens
            boundary = i
        budget_protect_count = len(result) - boundary
        protected_count = max(budget_protect_count, min_protect)
        prune_boundary = len(result) - protected_count
    else:
        prune_boundary = len(result) - protect_tail_count

    # Pass 1: Deduplicate identical tool results
    content_hashes: dict = {}
    for i in range(len(result) - 1, -1, -1):
        msg = result[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if isinstance(content, list):
            continue
        if not isinstance(content, str):
            continue
        if len(content) < 200:
            continue
        h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        if h in content_hashes:
            result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
            pruned += 1
        else:
            content_hashes[h] = (i, msg.get("tool_call_id", "?"))

    # Pass 2: Replace old tool results with informative summaries
    for i in range(prune_boundary):
        msg = result[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            continue
        if not isinstance(content, str):
            continue
        if not content or content == _PRUNED_PLACEHOLDER:
            continue
        if content.startswith("[Duplicate tool output"):
            continue
        if len(content) > 200:
            call_id = msg.get("tool_call_id", "")
            tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
            summary = _summarize_tool_result(tool_name, tool_args, content)
            result[i] = {**msg, "content": summary}
            pruned += 1

    # Pass 3: Truncate large tool_call arguments in assistant messages
    for i in range(prune_boundary):
        msg = result[i]
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        new_tcs = []
        modified = False
        for tc in msg["tool_calls"]:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
                if len(args) > 500:
                    new_args = _truncate_tool_call_args_json(args)
                    if new_args != args:
                        tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                        modified = True
            new_tcs.append(tc)
        if modified:
            result[i] = {**msg, "tool_calls": new_tcs}

    return result, pruned


# ── Serialization ────────────────────────────────────────────────────────────

_SERIALIZE_MAX = 6000
_SERIALIZE_HEAD = 4000
_SERIALIZE_TAIL = 1500


def _truncate_for_serialize(text: str) -> str:
    if len(text) > _SERIALIZE_MAX:
        return text[:_SERIALIZE_HEAD] + "\n...[truncated]...\n" + text[-_SERIALIZE_TAIL:]
    return text


def _serialize_turns(turns: List[Dict[str, Any]]) -> str:
    """Serialize conversation turns into labeled text for priority extraction."""
    parts = []
    for msg in turns:
        role = msg.get("role", "unknown")
        content = _truncate_for_serialize(msg.get("content") or "")

        if role == "tool":
            tool_id = msg.get("tool_call_id", "")
            parts.append(f"[TOOL RESULT {tool_id}]: {content}")
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if len(args) > 1500:
                            args = args[:1200] + "..."
                        tc_parts.append(f"  {name}({args})")
                content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
            parts.append(f"[ASSISTANT]: {content}")
            continue

        parts.append(f"[{role.upper()}]: {content}")

    return "\n\n".join(parts)


# ── Tool pair sanitization ──────────────────────────────────────────────────

def _sanitize_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs after compression."""
    surviving_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    cid = tc.get("call_id", "") or tc.get("id", "") or ""
                    if cid:
                        surviving_call_ids.add(cid)

    result_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # Remove orphaned results
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]

    # Add stub results for orphaned calls
    missing_results = surviving_call_ids - result_call_ids
    if missing_results:
        patched = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("call_id", "") or tc.get("id", "") or ""
                    else:
                        cid = getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": cid,
                        })
        messages = patched

    return messages


# ── Tail boundary helpers ────────────────────────────────────────────────────

def _align_boundary_forward(messages: List[Dict[str, Any]], idx: int) -> int:
    """Push boundary forward past any orphan tool results."""
    while idx < len(messages) and messages[idx].get("role") == "tool":
        idx += 1
    return idx


def _align_boundary_backward(messages: List[Dict[str, Any]], idx: int) -> int:
    """Pull boundary backward to avoid splitting tool_call/result groups."""
    if idx <= 0 or idx >= len(messages):
        return idx
    check = idx - 1
    while check >= 0 and messages[check].get("role") == "tool":
        check -= 1
    if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
        idx = check
    return idx


def _find_tail_cut_by_tokens(
    messages: List[Dict[str, Any]],
    head_end: int,
    token_budget: int,
) -> int:
    """Walk backward accumulating tokens until budget reached."""
    n = len(messages)
    min_tail = min(3, n - head_end - 1) if n - head_end > 1 else 0
    soft_ceiling = int(token_budget * 1.5)
    accumulated = 0
    cut_idx = n

    for i in range(n - 1, head_end - 1, -1):
        msg = messages[i]
        raw_content = msg.get("content") or ""
        content_len = _content_length_for_budget(raw_content)
        msg_tokens = content_len // _CHARS_PER_TOKEN + 10
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
                msg_tokens += len(args) // _CHARS_PER_TOKEN
        if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
            break
        accumulated += msg_tokens
        cut_idx = i

    fallback_cut = n - min_tail
    if cut_idx > fallback_cut:
        cut_idx = fallback_cut
    # Safety: if all messages fit in soft_ceiling, the tail swallows everything.
    # Force tail to at most 40% of total messages so middle has something to compress.
    max_tail_count = max(min_tail, int(n * 0.40))
    if n - cut_idx > max_tail_count:
        cut_idx = n - max_tail_count
    if cut_idx <= head_end:
        cut_idx = max(fallback_cut, head_end + 1)

    cut_idx = _align_boundary_backward(messages, cut_idx)

    # Ensure last user message is in tail — but never collapse below fallback
    fallback_floor = max(fallback_cut, head_end + 1)
    for i in range(n - 1, head_end - 1, -1):
        if messages[i].get("role") == "user":
            if i < cut_idx:
                cut_idx = max(i, fallback_floor)
            break

    return max(cut_idx, head_end + 1)


# ── Main engine class ───────────────────────────────────────────────────────

class PriorityCompressor(ContextEngine):
    """Deterministic priority-based context compression.

    No LLM calls. Uses priority-based line extraction to compress middle
    turns while preserving head (system prompt) and tail (recent context).
    """

    @property
    def name(self) -> str:
        return "priority_compressor"

    def __init__(
        self,
        model: str = "",
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 6,
        quiet_mode: bool = False,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
        max_summary_lines: int = _DEFAULT_MAX_LINES,
        max_chars_per_line: int = _DEFAULT_MAX_CHARS_PER_LINE,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.quiet_mode = quiet_mode
        self.max_summary_lines = max_summary_lines
        self.max_chars_per_line = max_chars_per_line

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        self._config_threshold_tokens = None  # Preserve config override across update_model() calls
        self.threshold_tokens = max(
            int(self.context_length * threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        # Override from config.yaml if set: context.threshold_tokens
        _config_threshold = _read_threshold_from_config()
        if _config_threshold is not None:
            self._config_threshold_tokens = _config_threshold
            self.threshold_tokens = max(_config_threshold, MINIMUM_CONTEXT_LENGTH)
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

        # Tail token budget: ~20% of threshold
        self.tail_token_budget = int(self.threshold_tokens * 0.20)

        if not quiet_mode:
            logger.info(
                "PriorityCompressor initialized: context=%d threshold=%d (%.0f%%) "
                "max_lines=%d max_chars=%d",
                self.context_length, self.threshold_tokens,
                threshold_percent * 100,
                max_summary_lines, max_chars_per_line,
            )

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.model = model
        self.context_length = context_length
        # Only recalculate threshold if no explicit config override was set.
        # This prevents update_model() from overwriting a user-set
        # context.threshold_tokens value when the context_length changes.
        if self._config_threshold_tokens is not None:
            self.threshold_tokens = max(self._config_threshold_tokens, MINIMUM_CONTEXT_LENGTH)
        else:
            self.threshold_tokens = max(
                int(context_length * self.threshold_percent),
                MINIMUM_CONTEXT_LENGTH,
            )
        self.tail_token_budget = int(self.threshold_tokens * 0.20)

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Compress messages using priority-based line extraction.

        Algorithm:
          1. Prune old tool results (cheap, no LLM)
          2. Protect head messages (system prompt + first exchange)
          3. Find tail boundary by token budget
          4. Serialize middle turns → extract priority summary
          5. Assemble: head + summary + tail
          6. Sanitize tool pairs
        """
        n_messages = len(messages)
        _min_for_compress = self.protect_first_n + 3 + 1
        if n_messages <= _min_for_compress:
            return messages

        display_tokens = current_tokens or self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: Prune old tool results
        messages, pruned_count = prune_old_tool_results(
            messages,
            protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )

        # Phase 2: Determine boundaries
        compress_start = self.protect_first_n
        compress_start = _align_boundary_forward(messages, compress_start)
        compress_end = _find_tail_cut_by_tokens(messages, compress_start, self.tail_token_budget)

        if compress_start >= compress_end:
            return messages

        turns_to_summarize = messages[compress_start:compress_end]

        if not self.quiet_mode:
            logger.info(
                "Priority compression: turns %d-%d (%d turns), "
                "protecting %d head + %d tail",
                compress_start + 1, compress_end,
                len(turns_to_summarize),
                compress_start, n_messages - compress_end,
            )

        # Phase 3: Serialize and extract priority summary
        serialized = _serialize_turns(turns_to_summarize)
        lines = serialized.split("\n")
        summary_body = extract_priority_summary(
            lines,
            max_lines=self.max_summary_lines,
            max_chars_per_line=self.max_chars_per_line,
        )

        if not summary_body.strip():
            summary_body = f"[{len(turns_to_summarize)} turns compressed — no extractable content]"

        summary = f"{_SUMMARY_PREFIX}\n{summary_body}"

        # Phase 4: Assemble compressed message list
        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                note = "[Note: Earlier conversation turns were compacted using priority-based extraction.]"
                existing = msg.get("content", "")
                if isinstance(existing, str) and note not in existing:
                    msg["content"] = existing + "\n\n" + note if existing else note
            compressed.append(msg)

        # Insert summary with appropriate role to avoid consecutive same-role
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"

        if last_head_role in ("assistant", "tool"):
            summary_role = "user"
        else:
            summary_role = "assistant"

        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped

        if summary_role == "user":
            summary += "\n\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"

        compressed.append({"role": summary_role, "content": summary})

        for i in range(compress_end, n_messages):
            compressed.append(messages[i].copy())

        self.compression_count += 1
        compressed = _sanitize_tool_pairs(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)
        saved_estimate = display_tokens - new_estimate

        if not self.quiet_mode:
            logger.info(
                "Priority compressed: %d -> %d messages (~%d tokens saved)",
                n_messages, len(compressed), saved_estimate,
            )

        return compressed
