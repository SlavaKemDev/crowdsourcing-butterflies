"""Use the local API queue with an isolated HW5 state directory."""

import json
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
import os
import time

ROOT = Path(__file__).resolve().parent
import platform_queue as control
from annotation import maximum_cost

control.RUN = ROOT / "run" / "private" / "api"
api = control.api
write_json = control.write_json
REQUESTER_ID = os.environ.get("HW5_REQUESTER_ID")
STATE_PATH = ROOT / "run" / "platform_state.json"


def state():
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(value):
    write_json(STATE_PATH, value)


def verify_account():
    account = api("/requester")
    if not REQUESTER_ID or account.get("id") != REQUESTER_ID:
        raise ValueError("The API account is not the verified HW5 team requester")
    return account


def budget_snapshot(reserve_rub=0):
    """Read-only lifetime upper bound for the team's fixed-price finite suites."""
    try:
        reserve = Decimal(str(reserve_rub))
    except InvalidOperation:
        raise ValueError("Reserve must be a finite nonnegative RUB amount") from None
    if not reserve.is_finite() or reserve < 0:
        raise ValueError("Reserve must be a finite nonnegative RUB amount")
    verify_account()
    current = state()
    if current.get("pending_action"):
        raise ValueError("Reconcile the pending platform action before checking the budget")
    known = {value["id"] for key, value in current.get("actions", {}).items()
             if key.startswith("pool-") and isinstance(value, dict) and isinstance(value.get("id"), str)}
    open_pools = api("/pools", params={"status": "OPEN"}, paginate=True)
    if open_pools.get("has_more") is not False or not isinstance(open_pools.get("items"), list):
        raise ValueError("Open-pool export is incomplete")
    for pool in open_pools["items"]:
        owner = pool.get("owner", {}).get("id")
        if not owner:
            raise ValueError("Cannot verify the owner of an open pool")
        if owner == REQUESTER_ID and pool["id"] not in known:
            raise ValueError("Unknown OPEN pool for this team requester: " + pool["id"])
    total, pools = Decimal(0), []
    for pool_id in sorted(known):
        pool = api("/pools/" + pool_id)
        if pool.get("owner", {}).get("id") != REQUESTER_ID:
            raise ValueError("Known pool has a different owner: " + pool_id)
        if any(pool.get(key) is not None for key in ("dynamic_pricing_config", "dynamic_overlap_config")):
            raise ValueError("Dynamic pricing or overlap is forbidden: " + pool_id)
        if pool.get("mixer_config"):
            raise ValueError("Fixed-suite budgeting does not support smart mixing: " + pool_id)
        for config in pool.get("quality_control", {}).get("configs", []):
            if any(rule.get("action", {}).get("type") == "CHANGE_OVERLAP" for rule in config.get("rules", [])):
                raise ValueError("Quality control can change the budgeted overlap: " + pool_id)
        for key, value in pool.get("defaults", {}).items():
            if "overlap" in key and (type(value) is not int or value < 0):
                raise ValueError("Pool overlap default must be a finite nonnegative integer")
        try:
            reward = Decimal(str(pool["reward_per_assignment"]))
        except (KeyError, InvalidOperation):
            raise ValueError("Missing or invalid reward in pool " + pool_id) from None
        if not reward.is_finite() or reward < 0:
            raise ValueError("Invalid reward in pool " + pool_id)
        exported = api("/task-suites", params={"pool_id": pool_id}, paginate=True)
        if exported.get("has_more") is not False or not isinstance(exported.get("items"), list):
            raise ValueError("Task-suite export is incomplete: " + pool_id)
        overlap, disabled, seen = 0, 0, set()
        for suite in exported["items"]:
            if not isinstance(suite.get("id"), str) or suite["id"] in seen:
                raise ValueError("Missing or duplicate task-suite identity")
            seen.add(suite["id"])
            if suite.get("pool_id") != pool_id or suite.get("infinite_overlap") is not False:
                raise ValueError("Wrong pool or infinite task-suite overlap")
            value = suite.get("overlap")
            if type(value) is not int or value < 0:
                raise ValueError("Task-suite overlap must be a finite nonnegative integer")
            overlap += value
            disabled += value == 0
        upper = maximum_cost([suite for suite in exported["items"] if suite["overlap"]], reward, 40)
        total += upper
        pools.append({"pool_id": pool_id, "status": pool.get("status"), "task_suites": len(seen),
                      "disabled_task_suites": disabled, "total_overlap": overlap,
                      "reward_per_assignment_rub": str(reward), "lifetime_upper_bound_rub": str(upper),
                      "suite_overlaps": [{"task_suite_id": suite["id"], "overlap": suite["overlap"]}
                                         for suite in exported["items"]]})
    combined = (total + reserve).quantize(Decimal("0.001"), rounding=ROUND_CEILING)
    if combined > Decimal(850):
        raise ValueError(f"Lifetime upper bound plus reserve exceeds 850 RUB: {combined}")
    return {"verified_requester_id": REQUESTER_ID, "checked_at_unix": time.time(), "pools": pools,
            "lifetime_upper_bound_rub": str(total), "reserve_rub": str(reserve),
            "upper_bound_with_reserve_rub": str(combined), "limit_rub": "850.000",
            "headroom_rub": str(Decimal(850) - combined), "within_limit": True,
            "scope": "Lifetime upper bound from total suite overlaps, including completed assignments; not actual expenditure."}


def mutate(key, path, method="POST", **kwargs):
    """Journal intent before mutation; uncertain outcomes require explicit reconciliation."""
    current = state()
    if current.get("pending_action"):
        raise RuntimeError("Reconcile pending_action with the API receipts before continuing")
    if key in current.get("actions", {}):
        raise ValueError("Action already completed; use the saved result")
    verify_account()
    current["pending_action"] = {"key": key, "path": path, "method": method, "time": time.time()}
    save_state(current)
    result = api(path, method, **kwargs)
    current.setdefault("actions", {})[key] = result
    current.pop("pending_action")
    save_state(current)
    return result
