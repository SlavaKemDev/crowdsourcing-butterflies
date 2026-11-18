"""Read-only crowd exports, review preparation, and independently checked golden set."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent
import crowd
from annotation import (maximum_cost, normalize_annotations, review_decision, task_suites,
                            validate_annotation, validate_golden)

PRIVATE = ROOT / "run" / "private"
STATUSES = ("ACTIVE", "SUBMITTED", "ACCEPTED", "REJECTED", "SKIPPED", "EXPIRED")


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    crowd.write_json(path, value)
    path.chmod(0o600)


def save_export(path, value):
    if path.exists():
        previous = path.read_bytes()
        archive = path.parent / "archive" / f"{path.stem}-{hashlib.sha256(previous).hexdigest()}.json"
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            archive.write_bytes(previous)
            archive.chmod(0o600)
        elif archive.read_bytes() != previous:
            raise ValueError("Existing export archive differs from its content hash")
    private_json(path, value)


def assignment_tasks_receipt(snapshot, assignment, request_id):
    """Verify the original same-assignment GET receipt before using any recovered tasks."""
    if not isinstance(request_id, str) or str(UUID(request_id)) != request_id:
        raise ValueError("Invalid assignment-task recovery request ID")
    raw = {kind: (PRIVATE / "api" / kind / f"{request_id}.json").read_bytes()
           for kind in ("requests", "responses", "receipts")}
    request, response, receipt = (json.loads(raw[kind]) for kind in ("requests", "responses", "receipts"))
    if not all(isinstance(value, dict) for value in (request, response, receipt)):
        raise ValueError("Malformed assignment-task source receipt")
    pool_id = snapshot["pool"]["id"]
    if (request.get("method") != "GET" or request.get("path") != "/assignments"
            or request.get("api_version") != "v1" or not isinstance(request.get("params"), dict)
            or request["params"].get("pool_id") != pool_id or response.get("ok") is not True
            or response.get("http_status") != 200 or response.get("request_id") != request_id
            or receipt.get("request_id") != request_id):
        raise ValueError("Recovery source is not a successful GET /assignments for this pool")
    exported = datetime.fromisoformat(snapshot["exported_at_utc"])
    started, finished = receipt.get("started_at"), response.get("finished_at")
    if (exported.tzinfo is None or any(type(value) not in (int, float) or not math.isfinite(value)
                                      for value in (started, finished))
            or not started <= finished <= exported.timestamp()):
        raise ValueError("Recovery source is not an earlier completed receipt")
    data = response.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("Recovery source lacks assignment items")
    matches = [row for row in data["items"] if isinstance(row, dict) and row.get("id") == assignment["id"]]
    if len(matches) != 1:
        raise ValueError("Recovery source must contain exactly this assignment once")
    source = matches[0]
    identity = {key: assignment.get(key) for key in ("id", "pool_id", "user_id", "task_suite_id")}
    if (any(not isinstance(value, str) or not value for value in identity.values())
            or identity["pool_id"] != pool_id or any(source.get(key) != value for key, value in identity.items())
            or assignment.get("status") != "ACCEPTED" or assignment.get("tasks") != []):
        raise ValueError("Recovery assignment identity or current empty-task state differs")
    tasks, solutions = source.get("tasks"), assignment.get("solutions")
    if (not isinstance(tasks, list) or not tasks or not isinstance(solutions, list) or not solutions
            or len(tasks) != len(solutions) or source.get("solutions") != solutions
            or any(not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"] for task in tasks)
            or len({task["id"] for task in tasks}) != len(tasks)):
        raise ValueError("Recovery task count, identities, or original solutions differ")
    return {"request_id": request_id, "request_sha256": hashlib.sha256(raw["requests"]).hexdigest(),
            "response_sha256": hashlib.sha256(raw["responses"]).hexdigest(),
            "receipt_sha256": hashlib.sha256(raw["receipts"]).hexdigest(), "identity": identity,
            "source_finished_at": finished, "tasks": tasks,
            "solutions_sha256": hashlib.sha256(json.dumps(solutions, sort_keys=True, ensure_ascii=False,
                                                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def recover_assignment_tasks(snapshot):
    missing = [row for row in snapshot["assignments"]["items"]
               if row.get("status") == "ACCEPTED" and row.get("tasks") == []
               and isinstance(row.get("solutions"), list) and row["solutions"]]
    recovered = {}
    if not missing:
        return recovered
    for path in sorted((PRIVATE / "api" / "requests").glob("*.json")):
        try:
            request = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (not isinstance(request, dict) or request.get("method") != "GET" or request.get("path") != "/assignments"
                or not isinstance(request.get("params"), dict) or request["params"].get("pool_id") != snapshot["pool"]["id"]):
            continue
        for assignment in missing:
            try:
                proof = assignment_tasks_receipt(snapshot, assignment, path.stem)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            previous = recovered.get(assignment["id"])
            if previous and previous["tasks"] != proof["tasks"]:
                raise ValueError("Conflicting original task receipts for the same assignment")
            if previous is None or (proof["source_finished_at"], proof["request_id"]) < (previous["source_finished_at"], previous["request_id"]):
                recovered[assignment["id"]] = proof
    return recovered


def pool_catalog(platform_state):
    actions = platform_state.get("actions", {})
    projects = {actions[key]["id"]: role for role in ("bbox", "review", "exam")
                if (key := "project-" + role) in actions}
    pools = {}
    for value in actions.values():
        if not isinstance(value, dict) or "reward_per_assignment" not in value:
            continue
        role = projects.get(value.get("project_id"))
        if role is None:
            continue
        if role == "exam":
            role = "bbox_exam" if "bbox" in value.get("private_name", "") else "review_exam"
        pools[value["id"]] = role
    return pools


def export_all():
    crowd.verify_account()
    platform_state = crowd.state()
    snapshots = []
    for pool_id, role in sorted(pool_catalog(platform_state).items()):
        pool = crowd.api("/pools/" + pool_id)
        if pool.get("owner", {}).get("id") != crowd.REQUESTER_ID:
            raise ValueError("Pool does not belong to the verified team requester")
        assignments = crowd.api("/assignments", params={
            "pool_id": pool_id, "status": ",".join(STATUSES)}, paginate=True)
        if assignments.get("has_more") is not False or not isinstance(assignments.get("items"), list):
            raise ValueError("Assignment export is incomplete")
        snapshot = {"pool": pool, "role": role, "assignments": assignments,
                    "exported_at_utc": datetime.now(timezone.utc).isoformat()}
        if role == "review":
            suites = crowd.api("/task-suites", params={"pool_id": pool_id}, paginate=True)
            if suites.get("has_more") is not False or not isinstance(suites.get("items"), list):
                raise ValueError("Review task-suite export is incomplete")
            snapshot["task_suites"] = suites
        recovery = recover_assignment_tasks(snapshot)
        if recovery:
            snapshot["recovered_assignment_tasks"] = recovery
        save_export(PRIVATE / "exports" / f"{pool_id}.json", snapshot)
        snapshots.append(snapshot)
    save_export(PRIVATE / "exports" / "index.json", {
        "pools": {snapshot["pool"]["id"]: snapshot["role"] for snapshot in snapshots},
        "action_keys": sorted(platform_state.get("actions", {}))})
    return snapshots


def load_exports():
    index = json.loads((PRIVATE / "exports" / "index.json").read_text())
    current = crowd.state()
    if (current.get("pending_action") or index["pools"] != pool_catalog(current)
            or index.get("action_keys") != sorted(current.get("actions", {}))):
        raise ValueError("Platform actions changed; run export before processing cached answers")
    return [json.loads((PRIVATE / "exports" / f"{pool_id}.json").read_text())
            for pool_id in sorted(index["pools"])]


def normalize_exports(snapshots):
    bbox, reviews, pending = [], [], []
    counts = defaultdict(Counter)
    earned = Decimal(0)
    fees = Decimal(0)
    cost_errors = 0
    recovered = {}
    seen_assignments, seen_records, authors = set(), set(), set()
    for snapshot in snapshots:
        role, pool_id = snapshot["role"], snapshot["pool"]["id"]
        counts[role].update({status: 0 for status in STATUSES})
        export = snapshot["assignments"]
        if export.get("has_more") is not False:
            raise ValueError("Cannot normalize a partial assignment export")
        recovery = snapshot.get("recovered_assignment_tasks", {})
        if not isinstance(recovery, dict) or set(recovery) - {row.get("id") for row in export["items"]}:
            raise ValueError("Recovery metadata has unknown assignment IDs")
        for assignment in export["items"]:
            assignment_id = assignment.get("id")
            if not isinstance(assignment_id, str) or assignment_id in seen_assignments:
                raise ValueError("Missing or duplicate assignment ID")
            seen_assignments.add(assignment_id)
            status, user = assignment.get("status"), assignment.get("user_id")
            if role == "bbox" and isinstance(user, str) and user:
                authors.add(user)
            counts[role][str(status)] += 1
            base = {"assignment_id": assignment_id, "pool_id": pool_id,
                    "user_id": user, "assignment_status": status, "role": role}
            if status not in STATUSES:
                pending.append({**base, "reason": "unknown_assignment_status"})
                continue
            if status == "ACCEPTED":
                try:
                    reward = Decimal(str(assignment["reward"]))
                    if not reward.is_finite() or reward < 0:
                        raise ValueError("Invalid reward")
                    earned += reward
                    fees += max(reward * Decimal("0.40"), Decimal("0.001")) if reward else Decimal(0)
                except (KeyError, InvalidOperation, ValueError):
                    cost_errors += 1
                    pending.append({**base, "reason": "invalid_accepted_reward"})
            if role not in {"bbox", "review"}:
                continue
            tasks, solutions = assignment.get("tasks"), assignment.get("solutions")
            if assignment_id in recovery:
                proof = recovery[assignment_id]
                try:
                    verified = assignment_tasks_receipt(snapshot, assignment, proof["request_id"])
                except (OSError, KeyError, TypeError, ValueError) as error:
                    raise ValueError("Untrusted assignment-task recovery source") from error
                if proof != verified:
                    raise ValueError("Assignment-task recovery metadata differs from its original receipt")
                tasks = verified["tasks"]
                recovered[assignment_id] = verified
                base["task_source_recovery"] = {key: verified[key] for key in ("request_id", "response_sha256")}
            if status in {"ACTIVE", "SKIPPED", "EXPIRED"}:
                pending.append({**base, "reason": "assignment_not_final",
                                "task_inputs": [task.get("input_values") for task in tasks or []]})
                continue
            if (not isinstance(tasks, list) or not tasks or not isinstance(solutions, list)
                    or len(tasks) != len(solutions) or not isinstance(user, str) or not user):
                pending.append({**base, "reason": "invalid_identity_or_tasks_solutions_mismatch"})
                continue
            for task, solution in zip(tasks, solutions):
                if isinstance(task, dict) and task.get("known_solutions"):
                    continue
                record = {**base, "submitted_at": assignment.get("submitted")}
                try:
                    task_id = task["id"]
                    inputs, outputs = task["input_values"], solution["output_values"]
                    photo_id = inputs["photo_id"]
                    if not isinstance(task_id, str) or not task_id or not isinstance(photo_id, str) or not photo_id.isdecimal() or int(photo_id) <= 0:
                        raise ValueError("Invalid task or photo identity")
                    record_id = assignment_id + ":" + task_id
                    if record_id in seen_records:
                        raise ValueError("Repeated task identity in assignment")
                    seen_records.add(record_id)
                    record.update(photo_id=int(photo_id), task_id=task_id,
                                  image=inputs.get("image"),
                                  comment=outputs.get("comment", ""))
                    if role == "bbox":
                        record.update(answer_id=assignment_id + ":" + task_id, status=outputs["status"],
                                      annotations=outputs["annotations"], boxes=normalize_annotations(outputs["annotations"]))
                        validate_annotation(record)
                        bbox.append(record)
                    else:
                        verdict, answer_id = outputs["verdict"], inputs["answer_id"]
                        if verdict not in {"accept", "reject", "uncertain", "load_error"} or not isinstance(answer_id, str) or not answer_id:
                            raise ValueError("Invalid review verdict or answer identity")
                        if verdict != "accept" and (not isinstance(record["comment"], str) or not record["comment"].strip()):
                            raise ValueError("Review issue requires an explanation")
                        record.update(review_id=assignment_id + ":" + task_id,
                                      answer_id=answer_id, verdict=verdict,
                                      reviewed_annotations=inputs.get("annotations"))
                        reviews.append(record)
                except (KeyError, TypeError, ValueError) as error:
                    pending.append({**record, "reason": "invalid_task_solution", "detail": str(error)})
    summary = {"assignment_counts": {role: dict(value) for role, value in sorted(counts.items())},
               "bbox_answers": len(bbox), "review_answers": len(reviews), "pending_records": len(pending),
               "cost_complete": cost_errors == 0,
               "recovered_assignments": len(recovered),
               "recovered_tasks": sum(len(proof["tasks"]) for proof in recovered.values()),
               "commission_percent": 40,
               "accepted_worker_reward_rub": str(earned.quantize(Decimal("0.001"))),
               "commission_on_accepted_rub": str(fees.quantize(Decimal("0.001"))),
               "accepted_reward_with_commission_rub": str((earned + fees).quantize(Decimal("0.001")))}
    return {"bbox": bbox, "reviews": reviews, "pending": pending, "bbox_author_ids": sorted(authors),
            "recovered_assignment_tasks": recovered, "summary": summary}


def audit_review_controls(snapshot, plan, pool_id, evidence):
    """Audit a frozen controls-v3/v4 wave; return an exclusion proposal, without writes."""
    if (not isinstance(snapshot, dict) or snapshot.get("role") != "review"
            or not isinstance(snapshot.get("pool"), dict) or not isinstance(pool_id, str)
            or not pool_id.strip() or snapshot["pool"].get("id") != pool_id
            or not isinstance(evidence, str) or not evidence.strip()
            or not isinstance(plan, dict) or plan.get("controls_per_page") != 2
            or plan.get("real_photos_per_page") != 4):
        raise ValueError("Control audit requires the explicit review pool, frozen plan, and evidence")
    version = plan.get("control_set_version", 3)
    if type(version) is not int or version not in (3, 4):
        raise ValueError("Unknown live control-set version")
    partial = plan.get("allow_partial_controls", False)
    if type(partial) is not bool or (partial and version != 4):
        raise ValueError("Partial control plans require an explicit v4 opt-in")
    overlaps = (1, 2, 3) if partial else (2,)
    real_counts = (4,) if version == 3 else (1, 2, 3, 4)
    exports = (snapshot.get("assignments"), snapshot.get("task_suites"))
    if any(not isinstance(value, dict) or value.get("has_more") is not False
           or not isinstance(value.get("items"), list) for value in exports):
        raise ValueError("Control audit requires complete assignment and task-suite exports")
    expected = {"control-v3-review-" + number: verdict
                for verdict, numbers in (("accept", ("01", "06", "08", "10")),
                                         ("reject", ("02", "03", "05", "07"))) for number in numbers}
    if version == 4:
        expected.update({"control-v4-moth-751516855": "accept", "control-v4-moth-757136849": "accept"})

    def user_identity(value):
        return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)

    def task_definition(task):
        inputs = task.get("input_values") if isinstance(task, dict) else None
        if (not isinstance(inputs, dict)
                or any(not isinstance(inputs.get(key), str) or not inputs[key].strip()
                       for key in ("photo_id", "image", "answer_id"))
                or not inputs["photo_id"].isdecimal() or int(inputs["photo_id"]) <= 0):
            raise ValueError("Invalid review task identity in control audit")
        normalize_annotations(inputs.get("annotations"))
        answer_id, known = inputs["answer_id"], task.get("known_solutions")
        verdict = expected.get(answer_id)
        if verdict:
            if (not isinstance(known, list) or len(known) != 1 or not isinstance(known[0], dict)
                    or set(known[0]) != {"output_values", "correctness_weight"}
                    or known[0]["output_values"] != {"verdict": verdict}
                    or type(known[0]["correctness_weight"]) not in (int, float)
                    or known[0]["correctness_weight"] != 1):
                raise ValueError("Trusted control has invalid known_solutions or required verdict")
        elif known not in (None, []) or answer_id.startswith("control-"):
            raise ValueError("Unknown or untrusted review control")
        return inputs, verdict

    planned = plan.get("task_suites")
    if not isinstance(planned, list) or not planned:
        raise ValueError("Frozen control plan requires task_suites")
    definitions, catalog = [], {}
    for page in planned:
        if (not isinstance(page, dict) or not isinstance(page.get("tasks"), list)
                or len(page["tasks"]) - 2 not in real_counts
                or type(page.get("overlap")) is not int or page["overlap"] not in overlaps
                or page.get("infinite_overlap") is not False):
            raise ValueError("Each planned page requires its allowed real tasks, two controls, and finite overlap")
        definition = [task_definition(task) for task in page["tasks"]]
        controls = [(inputs, verdict) for inputs, verdict in definition if verdict]
        real = [inputs for inputs, verdict in definition if not verdict]
        unavailable = page.get("unavailable_for", [])
        if (sorted(verdict for _, verdict in controls) != ["accept", "reject"] or len(real) not in real_counts
                or len({inputs["photo_id"] for inputs in real}) != len(real)
                or len({inputs["answer_id"] for inputs, _ in definition}) != len(definition)
                or not isinstance(unavailable, list) or any(not user_identity(user) for user in unavailable)
                or len(unavailable) != len(set(unavailable)) or definition in definitions):
            raise ValueError("Invalid or duplicate planned control page")
        definitions.append(definition)
        for inputs, verdict in controls:
            answer_id = inputs["answer_id"]
            if answer_id in catalog and catalog[answer_id] != (inputs, verdict):
                raise ValueError("The same trusted control has conflicting inputs")
            catalog[answer_id] = inputs, verdict
    if not partial and set(catalog) != set(expected):
        raise ValueError("Frozen plan must contain every trusted task for its control-set version")
    suites, matched_pages = {}, set()
    for suite in exports[1]["items"]:
        if (not isinstance(suite, dict) or not isinstance(suite.get("id"), str) or not suite["id"].strip()
                or suite["id"] in suites or suite.get("pool_id") != pool_id
                or not isinstance(suite.get("tasks"), list)):
            raise ValueError("Invalid or duplicate exported task-suite identity")
        definition = [task_definition(task) for task in suite["tasks"]]
        matches = [i for i, value in enumerate(definitions) if value == definition]
        if len(matches) != 1 or matches[0] in matched_pages:
            raise ValueError("Exported task-suite inputs or order differ from the frozen plan")
        page_index = matches[0]
        page = planned[page_index]
        task_ids = [task.get("id") for task in suite["tasks"]]
        unavailable = suite.get("unavailable_for", [])
        if (any(not isinstance(value, str) or not value.strip() for value in task_ids)
                or len(task_ids) != len(set(task_ids)) or type(suite.get("overlap")) is not int
                or suite["overlap"] != page["overlap"]
                or suite.get("infinite_overlap") is not False
                or not isinstance(unavailable, list) or any(not user_identity(user) for user in unavailable)
                or len(unavailable) != len(set(unavailable))
                or set(unavailable) != set(page.get("unavailable_for", []))):
            raise ValueError("Exported task-suite task identities or exclusions differ from the plan")
        matched_pages.add(page_index)
        suites[suite["id"]] = suite, page_index
    if len(matched_pages) != len(planned):
        raise ValueError("Task-suite export is missing frozen control pages")
    pages, seen_assignments, seen_page_users, failed_ids = [], set(), set(), []
    for assignment in exports[0]["items"]:
        if (not isinstance(assignment, dict) or not isinstance(assignment.get("id"), str)
                or not assignment["id"].strip() or assignment["id"] in seen_assignments):
            raise ValueError("Invalid or duplicate assignment identity in control audit")
        seen_assignments.add(assignment["id"])
        if assignment.get("status") != "ACCEPTED":
            continue
        suite_id, user = assignment.get("task_suite_id"), assignment.get("user_id")
        if (assignment.get("pool_id") != pool_id or not isinstance(suite_id, str)
                or suite_id not in suites or not user_identity(user)
                or (suite_id, user) in seen_page_users):
            raise ValueError("Invalid ACCEPTED assignment pool, suite, or independent reviewer identity")
        suite, page_index = suites[suite_id]
        if user in planned[page_index].get("unavailable_for", []):
            raise ValueError("An excluded reviewer completed this control page")
        tasks, solutions = assignment.get("tasks"), assignment.get("solutions")
        if (not isinstance(tasks, list) or len(tasks) != len(suite["tasks"]) or not isinstance(solutions, list)
                or len(solutions) != len(tasks) or any(not isinstance(task, dict) for task in tasks)
                or [task.get("id") for task in tasks] != [task["id"] for task in suite["tasks"]]
                or [task_definition(task) for task in tasks] != definitions[page_index]):
            raise ValueError("ACCEPTED tasks/solutions are partial or differ from exact task-suite order")
        seen_page_users.add((suite_id, user))
        controls, real_ids = [], []
        for index, (task, solution) in enumerate(zip(tasks, solutions)):
            outputs = solution.get("output_values") if isinstance(solution, dict) else None
            if (not isinstance(outputs, dict) or not isinstance(outputs.get("verdict"), str)
                    or outputs["verdict"] not in {"accept", "reject", "uncertain", "load_error"}
                    or not isinstance(outputs.get("comment", ""), str)):
                raise ValueError("ACCEPTED task lacks a valid required verdict")
            answer_id = task["input_values"]["answer_id"]
            if answer_id in expected:
                controls.append({"task_index": index, "task_id": task["id"], "answer_id": answer_id,
                                 "expected_verdict": expected[answer_id], "actual_verdict": outputs["verdict"],
                                 "passed": outputs["verdict"] == expected[answer_id]})
            else:
                if outputs["verdict"] != "accept" and not outputs.get("comment", "").strip():
                    raise ValueError("Real review issue lacks its required explanation")
                real_ids.append(assignment["id"] + ":" + task["id"])
        correct = sum(control["passed"] for control in controls)
        page = {"assignment_id": assignment["id"], "task_suite_id": suite_id, "planned_page_index": page_index,
                "user_id": user, "controls": controls, "correct_controls": correct,
                "total_controls": 2, "passed": correct == 2, "real_review_ids": real_ids}
        pages.append(page)
        if not page["passed"]:
            failed_ids.extend(real_ids)
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"pool_id": pool_id, "snapshot_sha256": digest(snapshot), "plan_sha256": digest(plan),
            "accepted_assignments": len(pages), "passed_assignments": sum(page["passed"] for page in pages),
            "failed_assignments": sum(not page["passed"] for page in pages), "pages": pages,
            "review_exclusions": {"reason": "At least one of two trusted in-page controls failed",
                                  "evidence": evidence, "review_ids": failed_ids,
                                  "evidence_details": [page for page in pages if not page["passed"]]}}


def manifest_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row.get("photo_id") for row in rows]
    if not rows or any(type(photo_id) is not int or photo_id <= 0 for photo_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest must have unique positive integer photo_ids")
    return {row["photo_id"]: row for row in rows}


def prepare_review(snapshots, normalized, manifest, pool_id):
    if not any(s["role"] == "review" and s["pool"]["id"] == pool_id for s in snapshots):
        raise ValueError("Target pool is not an exported HW5 review pool")
    authors, uploaded = set(), set()
    for snapshot in snapshots:
        if snapshot["role"] == "bbox":
            if snapshot["pool"].get("status") == "OPEN" or any(
                row.get("status") in {"ACTIVE", "SUBMITTED"} for row in snapshot["assignments"]["items"]
            ):
                raise ValueError("Close and finish the bbox wave before freezing its author cohort")
            authors.update(row["user_id"] for row in snapshot["assignments"]["items"] if row.get("user_id"))
        if snapshot["role"] == "review":
            if snapshot.get("task_suites", {}).get("has_more") is not False:
                raise ValueError("Need a complete review task-suite export to avoid duplicate uploads")
            for suite in snapshot["task_suites"]["items"]:
                uploaded.update(task.get("input_values", {}).get("answer_id") for task in suite.get("tasks", []))
    uploaded.update(row["answer_id"] for row in normalized["reviews"])
    tasks, manual, seen = [], [], set()
    for answer in sorted(normalized["bbox"], key=lambda row: (row["photo_id"], row["answer_id"])):
        if answer["assignment_status"] != "ACCEPTED":
            continue
        if answer["photo_id"] not in manifest:
            raise ValueError("Bbox photo_id is outside the frozen manifest")
        if answer["image"] != manifest[answer["photo_id"]]["image_url"]:
            raise ValueError("Bbox image URL differs from the frozen manifest")
        if answer["answer_id"] in seen:
            raise ValueError("Duplicate bbox answer_id")
        seen.add(answer["answer_id"])
        if answer["status"] in {"load_error", "uncertain"}:
            manual.append({"answer_id": answer["answer_id"], "photo_id": answer["photo_id"],
                           "reason": answer["status"]})
        elif answer["answer_id"] not in uploaded:
            tasks.append({"input_values": {"photo_id": str(answer["photo_id"]),
                          "image": manifest[answer["photo_id"]]["image_url"],
                          "answer_id": answer["answer_id"], "annotations": answer["annotations"]}})
    suites = task_suites(tasks, pool_id, 2, 2)
    return {"suites": suites, "author_ids_to_exclude": sorted(authors),
            "manual": manual, "already_uploaded_answers": len(uploaded - {None}), "new_review_tasks": len(tasks)}


def aggregate(normalized, manifest, expected_count=400, review_exclusions=None, adjudications=None,
              assisted_corrections=None):
    excluded_ids = set()
    if review_exclusions is not None:
        if (not isinstance(review_exclusions, dict)
                or not {"reason", "evidence", "review_ids"} <= set(review_exclusions)
                or any(not isinstance(review_exclusions.get(key), str) or not review_exclusions[key].strip()
                       for key in ("reason", "evidence"))):
            raise ValueError("Review exclusions require nonempty reason and evidence plus review_ids")
        ids = review_exclusions["review_ids"]
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("Excluded review_ids must be a list of nonempty strings")
        excluded_ids = set(ids)
        if len(excluded_ids) != len(ids):
            raise ValueError("Duplicate excluded review_id")
        known = [row.get("review_id") for row in normalized["reviews"] if row["assignment_status"] == "ACCEPTED"]
        if len(known) != len(set(known)):
            raise ValueError("Duplicate ACCEPTED review_id")
        if excluded_ids - set(known):
            raise ValueError("Unknown or non-ACCEPTED excluded review_id")
    excluded_reviews = [dict(row) for row in normalized["reviews"]
                        if row.get("review_id") in excluded_ids and row["assignment_status"] == "ACCEPTED"]
    accepted = {row["answer_id"]: row for row in normalized["bbox"] if row["assignment_status"] == "ACCEPTED"}
    candidate_answers, proposals, proposal_ids = dict(accepted), [], set()
    if assisted_corrections is not None:
        if (not isinstance(assisted_corrections, dict) or assisted_corrections.get("model_assisted") is not True
                or not isinstance(assisted_corrections.get("proposals"), list)
                or any(not isinstance(assisted_corrections.get(key), str) or not assisted_corrections[key].strip()
                       for key in ("reason", "evidence"))):
            raise ValueError("Assisted corrections require model_assisted=true, reason, evidence, and proposals")
        real_ids = {row["answer_id"] for row in normalized["bbox"]}
        real_ids.update(row["answer_id"] for row in normalized["pending"]
                        if row.get("role") == "bbox" and row.get("answer_id"))
        for proposal in assisted_corrections["proposals"]:
            if (not isinstance(proposal, dict) or type(proposal.get("photo_id")) is not int
                    or proposal.get("status") not in ("objects", "empty")
                    or any(not isinstance(proposal.get(key), str) or not proposal[key].strip()
                           for key in ("answer_id", "parent_answer_id", "reason"))):
                raise ValueError("Assisted proposal requires photo_id, answer_id, parent_answer_id, status, and reason")
            answer_id, photo_id = proposal["answer_id"], proposal["photo_id"]
            if answer_id in real_ids or answer_id in proposal_ids:
                raise ValueError("Assisted answer_id collides with a real bbox or another proposal")
            parent = accepted.get(proposal["parent_answer_id"])
            if (photo_id not in manifest or parent is None or parent["photo_id"] != photo_id
                    or parent.get("origin") == "model_assisted" or not isinstance(parent.get("user_id"), str)
                    or not parent["user_id"].strip()):
                raise ValueError("Assisted proposal requires an ACCEPTED human parent for the same manifest photo")
            if parent["image"] != manifest[photo_id]["image_url"]:
                raise ValueError("Assisted parent image URL differs from the manifest")
            validate_annotation({"status": proposal["status"], "boxes": proposal.get("boxes")})
            candidate = {key: proposal[key] for key in ("answer_id", "photo_id", "parent_answer_id", "status", "boxes", "reason")}
            candidate.update(origin="model_assisted", assignment_status="PROPOSED", user_id=None,
                             image=manifest[photo_id]["image_url"])
            candidate_answers[answer_id] = candidate
            proposals.append(candidate)
            proposal_ids.add(answer_id)
    reviews = defaultdict(list)
    issues = [row for row in normalized["pending"] if row.get("assignment_status") == "ACCEPTED"]
    for review in normalized["reviews"]:
        if review["assignment_status"] == "ACCEPTED":
            reviews[review["answer_id"]].append(review)
    for answer_id in reviews.keys() - candidate_answers.keys():
        issues.append({"answer_id": answer_id, "reason": "review_without_accepted_bbox"})
    decisions, approved, validated_reviews = [], defaultdict(list), {}
    for answer_id, answer in sorted(candidate_answers.items()):
        assisted = answer_id in proposal_ids
        included_reviews = [row for row in reviews[answer_id] if row["review_id"] not in excluded_ids]
        try:
            if answer["photo_id"] not in manifest:
                raise ValueError("Bbox photo is outside the manifest")
            if answer["image"] != manifest[answer["photo_id"]]["image_url"]:
                raise ValueError("Bbox image URL differs from the manifest")
            if any(review["photo_id"] != answer["photo_id"] for review in reviews[answer_id]):
                raise ValueError("Review photo_id differs from its original answer")
            if any(review["image"] != answer["image"] for review in reviews[answer_id]):
                raise ValueError("Review image URL differs from its original answer")
            if assisted and any(normalize_annotations(review.get("reviewed_annotations")) != answer["boxes"]
                                for review in reviews[answer_id]):
                raise ValueError("Assisted review annotations differ from proposed boxes")
            # Exclusion must not hide invalid identities, self-review, or repeated reviewers.
            decision = review_decision(answer, reviews[answer_id], assisted=assisted)
            if len(included_reviews) != len(reviews[answer_id]):
                decision = review_decision(answer, included_reviews, assisted=assisted)
            validated_reviews[answer_id] = included_reviews
            detail = None
        except ValueError as error:
            decision, detail = "manual", str(error)
        decision_record = {"answer_id": answer_id, "photo_id": answer["photo_id"], "decision": decision,
                           "review_ids": [row["review_id"] for row in included_reviews],
                           "excluded_review_ids": [row["review_id"] for row in reviews[answer_id]
                                                   if row["review_id"] in excluded_ids], "detail": detail}
        if assisted:
            decision_record["origin"] = "model_assisted"
        decisions.append(decision_record)
        if decision == "accept":
            approved[answer["photo_id"]].append(answer)
    selected, adjudication_audit = {}, []
    if adjudications is not None:
        if (not isinstance(adjudications, dict) or not isinstance(adjudications.get("answers"), list)
                or any(not isinstance(adjudications.get(key), str) or not adjudications[key].strip()
                       for key in ("reason", "evidence"))):
            raise ValueError("Adjudications require nonempty reason, evidence, and an answers list")
        for choice in adjudications["answers"]:
            if (not isinstance(choice, dict) or type(choice.get("photo_id")) is not int
                    or type(choice.get("allow_dissent", False)) is not bool
                    or type(choice.get("assistant_arbitration", False)) is not bool
                    or any(not isinstance(choice.get(key), str) or not choice[key].strip()
                           for key in ("answer_id", "reason"))):
                raise ValueError("Every adjudication requires integer photo_id, answer_id, and reason")
            photo_id, answer_id = choice["photo_id"], choice["answer_id"]
            if photo_id not in manifest:
                raise ValueError("Adjudicated photo_id is outside the manifest")
            if photo_id in selected:
                raise ValueError("Duplicate adjudicated photo_id")
            if answer_id not in candidate_answers or candidate_answers[answer_id]["photo_id"] != photo_id:
                raise ValueError("Adjudicated answer_id is unknown or belongs to another photo")
            candidate = next((row for row in approved[photo_id] if row["answer_id"] == answer_id), None)
            dissenting_ids = []
            arbitration = choice.get("assistant_arbitration", False)
            retained = validated_reviews.get(answer_id, [])
            independent_accepts = sum(review["verdict"] == "accept" for review in retained)
            if arbitration:
                if (answer_id not in proposal_ids or choice.get("operator_type") != "AI_assistant"
                        or not isinstance(choice.get("evidence"), str) or not choice["evidence"].strip()):
                    raise ValueError("Assistant arbitration requires an assisted proposal, AI_assistant, and per-photo evidence")
                parent = accepted[candidate_answers[answer_id]["parent_answer_id"]]
                # Membership means every identity/image/geometry check above passed, including excluded reviews.
                if (answer_id not in validated_reviews or independent_accepts < 1
                        or any(review["user_id"] == parent["user_id"] for review in reviews[answer_id])):
                    raise ValueError("Assistant arbitration requires a validated independent retained ACCEPTED accept")
                candidate = candidate_answers[answer_id]
                dissenting_ids = [review["review_id"] for review in retained if review["verdict"] != "accept"]
            elif candidate is None and choice.get("allow_dissent", False):
                if (candidate_answers[answer_id]["status"] in {"objects", "empty"}
                        and independent_accepts >= 2):
                    candidate = candidate_answers[answer_id]
                    dissenting_ids = [review["review_id"] for review in retained if review["verdict"] != "accept"]
            if candidate is None:
                raise ValueError("Adjudicated answer is not independently approved")
            selected[photo_id] = candidate
            audit = {"photo_id": photo_id, "selected_answer_id": answer_id,
                     "superseded_answer_ids": [row["answer_id"] for row in approved[photo_id]
                                               if row["answer_id"] != answer_id], "reason": choice["reason"]}
            if dissenting_ids:
                audit.update(accepted_with_dissent=True, dissenting_review_ids=dissenting_ids)
            if arbitration:
                audit.update(assistant_arbitration=True, operator_type="AI_assistant",
                             evidence=choice["evidence"], independent_human_accepts=independent_accepts,
                             human_verified=False)
            adjudication_audit.append(audit)
    golden, pending = [], []
    assisted_photos = {row["photo_id"] for row in proposals}
    for photo_id in sorted(manifest):
        candidates = [selected[photo_id]] if photo_id in selected else approved[photo_id]
        if photo_id in assisted_photos and photo_id not in selected:
            pending.append({"photo_id": photo_id,
                            "reason": "assisted_proposal_requires_explicit_adjudication",
                            "approved_answer_ids": [row["answer_id"] for row in candidates]})
        elif len(candidates) == 1:
            golden.append({"photo_id": photo_id, "boxes": candidates[0]["boxes"], "error": None})
        else:
            pending.append({"photo_id": photo_id, "reason": "no_independently_approved_answer" if not candidates else
                            "multiple_approved_candidates_need_adjudication",
                            "approved_answer_ids": [row["answer_id"] for row in candidates]})
    complete = len(manifest) == expected_count and not pending and not issues
    if complete:
        validate_golden(golden, manifest)
    return {"complete": complete, "manifest_photos": len(manifest), "expected_photos": expected_count,
            "golden": golden, "pending": pending, "issues": issues, "decisions": decisions,
            "review_exclusions": review_exclusions, "excluded_reviews": excluded_reviews,
            "adjudications": adjudications, "adjudication_audit": adjudication_audit,
            "assisted_corrections": assisted_corrections, "assisted_proposals": proposals,
            "summary": {"complete": complete, "manifest_photos": len(manifest), "expected_photos": expected_count,
                        "approved_photos": len(golden), "pending_photos": len(pending),
                        "excluded_reviews": len(excluded_reviews),
                        "answers_with_excluded_reviews": len({row["answer_id"] for row in excluded_reviews}),
                        "adjudicated_photos": len(selected),
                        "assistant_arbitrated_photos": sum(row.get("assistant_arbitration", False)
                                                           for row in adjudication_audit),
                        "accepted_with_dissent_photos": sum(row.get("accepted_with_dissent", False)
                                                            for row in adjudication_audit),
                        "assisted_proposals": len(proposals),
                        "assisted_approved": sum(row["answer_id"] in proposal_ids and row["decision"] == "accept"
                                                 for row in decisions),
                        "integrity_issues": len(issues), "decision_counts": dict(Counter(row["decision"] for row in decisions))}}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "export", "prepare-review", "aggregate"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.jsonl")
    parser.add_argument("--pool-id", help="Target review pool; defaults to the original HW5 review pool")
    parser.add_argument("--review-exclusions", type=Path, help="Explicit review exclusion audit JSON for aggregate")
    parser.add_argument("--adjudications", type=Path, help="Explicit independently approved answer choices for aggregate")
    parser.add_argument("--assisted-corrections", type=Path, help="Explicit model-assisted proposals for aggregate")
    args = parser.parse_args()
    if args.review_exclusions and args.command != "aggregate":
        parser.error("--review-exclusions is only valid for aggregate")
    if args.adjudications and args.command != "aggregate":
        parser.error("--adjudications is only valid for aggregate")
    if args.assisted_corrections and args.command != "aggregate":
        parser.error("--assisted-corrections is only valid for aggregate")
    exclusions = json.loads(args.review_exclusions.read_text()) if args.review_exclusions else None
    if args.review_exclusions and not isinstance(exclusions, dict):
        raise ValueError("Review exclusions file must contain a metadata object")
    adjudications = json.loads(args.adjudications.read_text()) if args.adjudications else None
    if args.adjudications and not isinstance(adjudications, dict):
        raise ValueError("Adjudications file must contain a metadata object")
    assisted = json.loads(args.assisted_corrections.read_text()) if args.assisted_corrections else None
    if args.assisted_corrections and not isinstance(assisted, dict):
        raise ValueError("Assisted corrections file must contain a metadata object")
    snapshots = export_all() if args.command in {"status", "export"} else load_exports()
    normalized = normalize_exports(snapshots)
    private_json(PRIVATE / "normalized.json", normalized)
    public = dict(normalized["summary"])
    if args.command == "prepare-review":
        pool_id = args.pool_id or crowd.state()["actions"]["pool-review"]["id"]
        plan = prepare_review(snapshots, normalized, manifest_rows(args.manifest), pool_id)
        private_json(PRIVATE / "prepared" / f"review-{pool_id}.json", plan)
        public.update(new_review_tasks=plan["new_review_tasks"], new_review_pages=len(plan["suites"]),
                      authors_to_exclude=len(plan["author_ids_to_exclude"]), manual_cases=len(plan["manual"]),
                      new_review_maximum_cost_rub=str(maximum_cost(plan["suites"], ".50", 40)))
    if args.command == "aggregate":
        result = aggregate(normalized, manifest_rows(args.manifest), review_exclusions=exclusions,
                           adjudications=adjudications, assisted_corrections=assisted)
        private_json(PRIVATE / "aggregation.json", result)
        public.update(result["summary"])
        if result["complete"]:
            destination = ROOT / "data" / "golden.jsonl"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["golden"]))
            temporary.replace(destination)
    crowd.write_json(ROOT / "run" / "crowd_summary.json", public)
    print(json.dumps(public, ensure_ascii=False))
    return 2 if args.command == "aggregate" and not public["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
