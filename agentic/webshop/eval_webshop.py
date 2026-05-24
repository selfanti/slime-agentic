#!/usr/bin/env python3
"""
WebShop Standalone Evaluation
==============================
Forward inference only; no training logic. Runs an LLM agent through the
WebShop text-mode Gym environment and computes reward per session.

Usage (server already running):
    python eval_webshop.py \
        --tokenizer /data/model/qwen25_7b/ \
        --server-url http://127.0.0.1:30000/generate \
        --output eval_results.json

Usage (auto-launch SGLang server):
    python eval_webshop.py \
        --model /data/model/qwen25_7b/ \
        --start-servers --tp 4 \
        --output eval_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_AGENTIC_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _AGENTIC_DIR.parent

for _p in (str(_SCRIPT_DIR), str(_AGENTIC_DIR / "agentflow")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.llm_engine import SGLangEngine  # noqa: E402
import slime.utils.http_utils as _http_utils  # noqa: E402

# WebShop env source
_WEBSHOP_DIR = _PROJECT_ROOT / "webshop"
if str(_WEBSHOP_DIR) not in sys.path:
    sys.path.insert(0, str(_WEBSHOP_DIR))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_webshop")


# ── HTTP client init (same as eval_agentflow.py) ───────────────────────────────
def _init_http_client(concurrency: int = 256) -> None:
    if _http_utils._http_client is None:
        _http_utils._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=concurrency),
            timeout=httpx.Timeout(None),
        )


# ── SGLang server management ───────────────────────────────────────────────────
def _wait_for_server(url: str, timeout: int = 300, interval: int = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(interval)
    return False


class SGLangServer:
    """Launch and manage a SGLang HTTP service in a subprocess."""

    def __init__(
        self,
        model_path: str,
        port: int,
        tp: int,
        mem_fraction: float,
        ctx_len: int,
        extra_args: list[str] | None = None,
    ):
        self.port = port
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--tp", str(tp),
            "--mem-fraction-static", str(mem_fraction),
            "--context-length", str(ctx_len),
            "--trust-remote-code",
        ] + (extra_args or [])
        logger.info("Starting SGLang server (port=%d): %s", port, " ".join(cmd))
        self._proc = subprocess.Popen(cmd)

    def wait_ready(self, timeout: int = 300) -> bool:
        url = f"http://127.0.0.1:{self.port}/health"
        logger.info("Waiting for port=%d (up to %ds)…", self.port, timeout)
        ok = _wait_for_server(url, timeout=timeout)
        if ok:
            logger.info("port=%d ready.", self.port)
        else:
            logger.error("port=%d not ready within %ds.", self.port, timeout)
        return ok

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ── Environment pool ───────────────────────────────────────────────────────────
# We share one SimServer (heavy: loads products & search engine) across multiple
# WebAgentTextEnv instances. Each env gets its own SimBrowser so sessions are
# independent and safe for interleaved async access.

def _create_env_pool(
    pool_size: int,
    observation_mode: str,
    num_products: int | None,
    limit_goals: int,
) -> list:
    """Create *pool_size* WebAgentTextEnv instances sharing one SimServer."""
    import gym
    from web_agent_site.envs.web_agent_text_env import SimServer

    base_url = "http://127.0.0.1:3000"
    from web_agent_site.utils import DEFAULT_FILE_PATH

    logger.info("Loading WebShop data (products, search engine, goals)…")
    server = SimServer(
        base_url=base_url,
        file_path=DEFAULT_FILE_PATH,
        limit_goals=limit_goals,
        num_products=num_products,
    )
    logger.info("SimServer loaded %d goals.", len(server.goals))

    envs = []
    for _ in range(pool_size):
        # Import here so the gym registration happens first
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        env = WebAgentTextEnv(
            observation_mode=observation_mode,
            server=server,
        )
        envs.append(env)
    return envs


# ── Prompt building ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a web shopping assistant navigating an online store. "
    "Your goal is to find and purchase a product that best matches the given instruction.\n\n"
    "On each turn you will see the current page content and a list of available actions. "
    "You must choose exactly ONE action:\n"
    "- search[keywords]: Search for products using keywords\n"
    "- click[button name]: Click on a button, link, or option\n\n"
    "Strategy:\n"
    "1. Start by searching for relevant products using key terms from the instruction.\n"
    "2. Browse search results and click on promising products.\n"
    "3. Check product details (click Description, Features) to verify the match.\n"
    "4. Select the correct options (color, size, etc.) if needed.\n"
    "5. Click \"Buy Now\" when you are confident the product matches.\n\n"
    "Output ONLY your action in the exact format: search[query] or click[name]. "
    "Do not output anything else."
)


def _build_messages(
    instruction: str,
    observation: str,
    available_actions: dict,
    prev_actions: list[str] | None = None,
    max_obs_chars: int = 4000,
) -> list[dict[str, str]]:
    """Build chat messages for one WebShop step."""
    # Truncate observation if too long
    obs = observation
    if len(obs) > max_obs_chars:
        obs = obs[:max_obs_chars] + "\n... (truncated)"

    parts: list[str] = []
    if prev_actions:
        parts.append("Previous actions:")
        for i, a in enumerate(prev_actions[-10:], 1):
            parts.append(f"  {i}. {a}")
        parts.append("")

    parts.append(f"Instruction: {instruction}")
    parts.append(f"\nCurrent page:\n{obs}\n")

    avail_lines: list[str] = []
    if available_actions.get("has_search_bar"):
        avail_lines.append("- search[...] to search for products")
    clickables = available_actions.get("clickables", [])
    if clickables:
        # Show clickable items in a compact format
        shown = clickables[:30]
        avail_lines.append("- Clickable items: " + ", ".join(f'"{c}"' for c in shown))
        if len(clickables) > 30:
            avail_lines.append(f"  ... and {len(clickables) - 30} more items")
    parts.append("Available actions:\n" + "\n".join(avail_lines))
    parts.append("\nWhat is your next action?")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ── Action parsing ─────────────────────────────────────────────────────────────

# Patterns for extracting actions from LLM output
_RE_SEARCH = re.compile(r"search\[(.+?)\]", re.IGNORECASE)
_RE_CLICK = re.compile(r"click\[(.+?)\]", re.IGNORECASE)


def _parse_action(response: str, available_actions: dict) -> str | None:
    """Parse the LLM response into a valid WebShop action string.

    Returns None if no valid action could be extracted.
    """
    response = response.strip()
    clickables = [c.lower() for c in available_actions.get("clickables", [])]

    # --- Try structured format first: search[...] / click[...] ---
    m = _RE_SEARCH.search(response)
    if m and available_actions.get("has_search_bar"):
        return f"search[{m.group(1).strip()}]"

    m = _RE_CLICK.search(response)
    if m:
        arg = m.group(1).strip()
        arg_lower = arg.lower()
        # Exact match
        for orig, orig_lower in zip(available_actions.get("clickables", []), clickables):
            if orig_lower == arg_lower:
                return f"click[{orig}]"
        # Substring / partial match
        for orig, orig_lower in zip(available_actions.get("clickables", []), clickables):
            if arg_lower in orig_lower or orig_lower in arg_lower:
                return f"click[{orig}]"
        # Return click with original arg (env will handle invalid gracefully)
        return f"click[{arg}]"

    # --- Natural language fallback ---
    nl_search = re.search(
        r"(?:search|find|look\s+for)\s+(?:for\s+)?[\"']?(.+?)[\"']?\s*$",
        response, re.IGNORECASE,
    )
    if nl_search and available_actions.get("has_search_bar"):
        return f"search[{nl_search.group(1).strip()}]"

    nl_click = re.search(
        r"(?:click|select|choose|press)\s+(?:on\s+)?[\"']?(.+?)[\"']?\s*$",
        response, re.IGNORECASE,
    )
    if nl_click:
        target = nl_click.group(1).strip().lower()
        for orig, orig_lower in zip(available_actions.get("clickables", []), clickables):
            if target in orig_lower or orig_lower in target:
                return f"click[{orig}]"

    # --- Last resort: if search bar available, use last line as query ---
    if available_actions.get("has_search_bar"):
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        if lines:
            query = re.sub(r"^(?:action|next|i\s*(?:will|'ll))\s*:\s*", "", lines[-1], flags=re.IGNORECASE)
            return f"search[{query}]"

    return None


# ── Single session evaluation ──────────────────────────────────────────────────

async def _eval_one_session(
    env,
    session_idx: int,
    engine: SGLangEngine,
    max_steps: int,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> dict:
    """Run one WebShop session from start to finish."""
    async with semaphore:
        try:
            obs, _ = env.reset(session=session_idx)
            instruction = env.get_instruction_text()
        except Exception as exc:
            logger.warning("[%d/%d] session=%d reset error: %s", idx + 1, total, session_idx, exc)
            return {
                "idx": idx, "session": session_idx, "instruction": "",
                "reward": 0.0, "num_steps": 0, "actions": [], "error": str(exc),
            }

        prev_actions: list[str] = []
        response = ""
        final_reward = 0.0

        for step in range(max_steps):
            available = env.get_available_actions()
            messages = _build_messages(instruction, obs, available, prev_actions)

            try:
                out = await engine.generate(messages)
                response = out.response.strip()
            except Exception as exc:
                logger.warning(
                    "[%d/%d] session=%d step=%d LLM error: %s",
                    idx + 1, total, session_idx, step, exc,
                )
                break

            action = _parse_action(response, available)
            if action is None:
                logger.warning(
                    "[%d/%d] session=%d step=%d invalid action from: %.80s",
                    idx + 1, total, session_idx, step, response,
                )
                break

            prev_actions.append(action)
            obs, reward, done, _info = env.step(action)
            final_reward = reward

            if done:
                break

        logger.info(
            "[%d/%d] session=%d reward=%.3f steps=%d",
            idx + 1, total, session_idx, final_reward, len(prev_actions),
        )

        return {
            "idx": idx,
            "session": session_idx,
            "instruction": instruction,
            "reward": final_reward,
            "num_steps": len(prev_actions),
            "actions": prev_actions,
            "final_output": response,
        }


# ── Batch evaluation ───────────────────────────────────────────────────────────

async def run_eval(
    env_pool: list,
    engine: SGLangEngine,
    session_indices: list[int],
    concurrency: int,
    max_steps: int,
) -> list[dict]:
    """Evaluate all sessions concurrently, distributing across env pool."""
    _init_http_client(concurrency=concurrency * 4)
    semaphore = asyncio.Semaphore(concurrency)
    total = len(session_indices)

    tasks = [
        _eval_one_session(
            env=env_pool[i % len(env_pool)],
            session_idx=session_idx,
            engine=engine,
            max_steps=max_steps,
            semaphore=semaphore,
            idx=i,
            total=total,
        )
        for i, session_idx in enumerate(session_indices)
    ]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["idx"])


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WebShop standalone evaluation (forward inference only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / Tokenizer
    model_grp = p.add_argument_group("Model configuration")
    model_grp.add_argument("--model", default=None,
                           help="HF model path (required when --start-servers is used)")
    model_grp.add_argument("--tokenizer", default=None,
                           help="HF tokenizer path (defaults to --model)")

    # Server connection
    srv_grp = p.add_argument_group("Server connection")
    srv_grp.add_argument("--server-url", default="http://127.0.0.1:30000/generate",
                         help="SGLang /generate URL")

    # Auto-launch servers
    auto_grp = p.add_argument_group("Auto-launch SGLang server")
    auto_grp.add_argument("--start-servers", action="store_true",
                          help="Auto-launch SGLang server (requires --model)")
    auto_grp.add_argument("--port", type=int, default=30000, help="Server port")
    auto_grp.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    auto_grp.add_argument("--mem-fraction", type=float, default=0.7)
    auto_grp.add_argument("--ctx-len", type=int, default=32768)

    # Evaluation data
    data_grp = p.add_argument_group("Evaluation data")
    data_grp.add_argument("--num-sessions", type=int, default=None,
                          help="Number of sessions to evaluate (default: all goals)")
    data_grp.add_argument("--session-start", type=int, default=0,
                          help="Starting session index")
    data_grp.add_argument("--num-products", type=int, default=None,
                          help="Number of products to load (None = all)")
    data_grp.add_argument("--limit-goals", type=int, default=-1,
                          help="Limit number of goals (-1 = no limit)")
    data_grp.add_argument("--observation-mode", default="text_rich",
                          choices=["html", "text", "text_rich", "url"])

    # Sampling parameters
    samp_grp = p.add_argument_group("Sampling parameters")
    samp_grp.add_argument("--temperature", type=float, default=0.7)
    samp_grp.add_argument("--top-p", type=float, default=0.95)
    samp_grp.add_argument("--max-new-tokens", type=int, default=512,
                          help="Max tokens for LLM action generation")

    # Eval control
    eval_grp = p.add_argument_group("Evaluation control")
    eval_grp.add_argument("--concurrency", type=int, default=16,
                          help="Maximum concurrent sessions")
    eval_grp.add_argument("--max-steps", type=int, default=20,
                          help="Maximum steps per session")

    # Output
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument("--output", default="eval_webshop_results.json",
                         help="Path to save evaluation results")
    out_grp.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer_path = args.tokenizer or args.model
    if not tokenizer_path:
        logger.error("Please provide --tokenizer or --model.")
        sys.exit(1)
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # ── SGLang servers ─────────────────────────────────────────────────────────
    servers: list[SGLangServer] = []
    server_url = args.server_url

    if args.start_servers:
        if not args.model:
            logger.error("--start-servers requires --model.")
            sys.exit(1)
        srv = SGLangServer(args.model, args.port, args.tp, args.mem_fraction, args.ctx_len)
        servers.append(srv)
        server_url = f"http://127.0.0.1:{args.port}/generate"
        if not srv.wait_ready(timeout=300):
            for s in servers:
                s.stop()
            sys.exit(1)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    engine = SGLangEngine(
        url=server_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
    )

    # ── Create env pool ────────────────────────────────────────────────────────
    pool_size = min(args.concurrency, 16)
    logger.info("Creating env pool (size=%d)…", pool_size)
    env_pool = _create_env_pool(
        pool_size=pool_size,
        observation_mode=args.observation_mode,
        num_products=args.num_products,
        limit_goals=args.limit_goals,
    )

    num_goals = len(env_pool[0].server.goals)
    logger.info("Total goals available: %d", num_goals)

    end_idx = num_goals if args.num_sessions is None else min(
        args.session_start + args.num_sessions, num_goals,
    )
    session_indices = list(range(args.session_start, end_idx))
    logger.info(
        "Evaluating %d sessions (concurrency=%d, max_steps=%d)",
        len(session_indices), args.concurrency, args.max_steps,
    )

    # ── Run evaluation ─────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        results = asyncio.run(run_eval(
            env_pool=env_pool,
            engine=engine,
            session_indices=session_indices,
            concurrency=args.concurrency,
            max_steps=args.max_steps,
        ))
    finally:
        for srv in servers:
            srv.stop()

    elapsed = time.time() - t0

    # ── Aggregate results ──────────────────────────────────────────────────────
    rewards = [r["reward"] for r in results]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    n_perfect = sum(1 for r in rewards if r >= 1.0)
    n_good = sum(1 for r in rewards if 0.5 <= r < 1.0)
    n_partial = sum(1 for r in rewards if 0 < r < 0.5)
    n_zero = sum(1 for r in rewards if r <= 0.0)
    errors = [r for r in results if "error" in r]

    output_data = {
        "avg_reward": round(avg_reward, 4),
        "num_sessions": len(results),
        "perfect": n_perfect,
        "good": n_good,
        "partial": n_partial,
        "zero": n_zero,
        "elapsed_seconds": round(elapsed, 2),
        "details": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))

    # ── Print summary ──────────────────────────────────────────────────────────
    n = len(results)
    print("\n" + "=" * 60)
    print("WebShop Evaluation Results")
    print("=" * 60)
    print(f"  Sessions:    {n}")
    print(f"  Avg Reward:  {avg_reward:.4f}")
    print(f"  Perfect (1.0):      {n_perfect:>4}  ({100 * n_perfect / n:.1f}%)" if n else "")
    print(f"  Good    (0.5-1.0):  {n_good:>4}  ({100 * n_good / n:.1f}%)" if n else "")
    print(f"  Partial (0-0.5):    {n_partial:>4}  ({100 * n_partial / n:.1f}%)" if n else "")
    print(f"  Zero    (0.0):      {n_zero:>4}  ({100 * n_zero / n:.1f}%)" if n else "")
    if errors:
        print(f"  Errors:             {len(errors):>4}")
    print(f"  Elapsed:     {elapsed:.1f}s")
    print("=" * 60)

    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
