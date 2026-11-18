"""HW5 project payloads and local annotation/review checks; no network calls."""

from decimal import Decimal, ROUND_CEILING
import json
import math


STATUS_OPTIONS = [("objects", "Есть бабочки"), ("empty", "Бабочек нет"),
                  ("load_error", "Ошибка загрузки"), ("uncertain", "Нужен разбор")]
VERDICT_OPTIONS = [("accept", "Принять разметку"), ("reject", "Отклонить разметку"),
                   ("uncertain", "Нужен разбор"), ("load_error", "Ошибка загрузки")]
EXAM_QUESTIONS = {
    "bbox": [
        ("Нужно ли выделять взрослую ночную бабочку?", "yes"),
        ("Можно ли обвести двух соприкасающихся бабочек одной общей рамкой?", "no"),
        ("Нужно ли включать видимые усики и лапки в рамку?", "yes"),
        ("Нужно ли достраивать скрытое листом крыло?", "no"),
        ("Можно ли ответить 'Бабочек нет', если фотография не загрузилась?", "no"),
        ("Нужно ли выделять гусеницу, если взрослых бабочек в кадре нет?", "no"),
        ("Может ли на фотографии цветка быть бабочка, которую нужно выделить?", "yes"),
        ("Нужно ли выделить видимую часть бабочки, обрезанной краем фотографии?", "yes"),
        ("Нужно ли отдельно обводить тень бабочки?", "no"),
        ("Если объект нельзя уверенно опознать, нужен ли отдельный разбор сомнения?", "yes"),
    ],
    "review": [
        ("Нужно ли осматривать весь кадр, включая участки вне предложенных рамок?", "yes"),
        ("Можно ли проверять собственную первичную разметку?", "no"),
        ("Если две бабочки выделены одной общей рамкой, нужно ли отклонить ответ?", "yes"),
        ("Если одна взрослая ночная бабочка пропущена, нужно ли отклонить ответ?", "yes"),
        ("Можно ли принять пустую разметку, если весь кадр проверен и взрослых бабочек нет?", "yes"),
        ("Можно ли принять разметку, когда изображение не загрузилось?", "no"),
        ("Нужно ли отклонить рамку, выделяющую гусеницу вместо взрослой бабочки?", "yes"),
        ("Нужно ли требовать дорисовки крыла, скрытого листом?", "no"),
        ("Можно ли принять две повторные рамки вокруг одной бабочки?", "no"),
        ("Если объект неоднозначен, нужно ли выбрать 'Нужен разбор' и пояснить сомнение?", "yes"),
    ],
}


def choice(path, label, options):
    return {"type": "field.button-radio-group", "label": label,
            "data": {"type": "data.output", "path": path},
            "options": [{"value": value, "label": text} for value, text in options],
            "validation": {"type": "condition.required"}}


def build_project(role, instructions_html):
    """Return POST /projects payload. Preview in the real service before publishing."""
    if role not in {"bbox", "review", "exam"}:
        raise ValueError("Unknown project role")
    inputs = {"photo_id": {"type": "string", "required": True},
              "image": {"type": "url", "required": True}}
    if role == "exam":
        inputs = {"question": {"type": "string", "required": True},
                  "image": {"type": "url", "required": False},
                  "annotations": {"type": "json", "required": False}}
        outputs = {"answer": {"type": "string", "required": True,
                              "allowed_values": ["yes", "no"]}}
        items = [{"type": "view.text", "content": {"type": "data.input", "path": "question"}},
                 {"type": "helper.if", "condition": {"type": "condition.empty", "data": {
                     "type": "data.input", "path": "image", "default": ""}},
                  "then": {"type": "view.text", "content": "Вопрос по инструкции"},
                  "else": {"type": "field.image-annotation", "disabled": True, "fullHeight": True,
                           "image": {"type": "data.input", "path": "image"},
                           "data": {"type": "data.internal", "path": "exam_annotations", "default": {
                               "type": "data.input", "path": "annotations", "default": []}},
                           "shapes": {"rectangle": True, "polygon": False, "point": False}}},
                 choice("answer", "Выберите ответ по инструкции", [("yes", "Да"), ("no", "Нет")])]
        name = "HW5: экзамен по правилам разметки и проверки бабочек"
    else:
        annotation_data = {"type": "data.output", "path": "annotations", "default": []}
        if role == "review":
            inputs.update({"answer_id": {"type": "string", "required": True},
                           "annotations": {"type": "json", "required": True}})
            annotation_data = {"type": "data.internal", "path": "shown_annotations",
                               "default": {"type": "data.input", "path": "annotations"}}
        editor = {"type": "field.image-annotation", "data": annotation_data,
                  "image": {"type": "data.input", "path": "image"},
                  "shapes": {"rectangle": True, "polygon": False, "point": False},
                  "fullHeight": True, "disabled": role == "review"}
        items = [{"type": "view.text", "content": (
            "Осмотрите весь кадр. Выделите каждую взрослую бабочку отдельно, включая ночных."
            if role == "bbox" else
            'Проверяйте правильность разметки. Если бабочек нет и рамок нет, нажмите "Принять разметку". '
            "Осмотрите весь кадр: ищите пропуски, лишние рамки и неточные границы.")}, editor]
        outputs = {"comment": {"type": "string", "required": False}}
        if role == "bbox":
            inputs["correction_hint"] = {"type": "string", "required": False}
            items.insert(1, {"type": "view.text", "content": {
                "type": "data.input", "path": "correction_hint", "default": ""}})
            outputs.update({"status": {"type": "string", "required": True,
                                       "allowed_values": [value for value, _ in STATUS_OPTIONS]},
                            "annotations": {"type": "json", "required": True}})
            status_is = lambda value: {"type": "condition.equals", "data": {
                "type": "data.output", "path": "status"}, "to": value}
            editor["validation"] = {"type": "condition.any", "conditions": [
                {"type": "condition.all", "conditions": [status_is("objects"), {
                    "type": "condition.schema", "data": annotation_data,
                    "schema": {"type": "array", "minItems": 1}}]},
                {"type": "condition.all", "conditions": [{"type": "condition.any", "conditions": [
                    status_is("empty"), status_is("load_error")]},
                    {"type": "condition.schema", "data": annotation_data,
                     "schema": {"type": "array", "maxItems": 0}}]}, status_is("uncertain")],
                "hint": "При наличии бабочек нарисуйте рамки; для пустого кадра или ошибки загрузки удалите рамки."}
            items.append(choice("status", "Результат просмотра", STATUS_OPTIONS))
        else:
            outputs.update({"verdict": {"type": "string", "required": True,
                                        "allowed_values": [value for value, _ in VERDICT_OPTIONS]}})
            items.append(choice("verdict", "Верна ли предложенная разметка?", VERDICT_OPTIONS))
        comment = {"type": "field.textarea", "label": "Причины ошибок или сомнений и их место в кадре",
                   "data": {"type": "data.output", "path": "comment"},
                   "placeholder": "Например: пропущена маленькая бабочка справа внизу"}
        if role == "review":
            comment["validation"] = {"type": "condition.any", "conditions": [
                {"type": "condition.equals", "data": {"type": "data.output", "path": "verdict"},
                 "to": "accept"}, {"type": "condition.required", "data": comment["data"]}],
                "hint": "Опишите причину отклонения, сомнения или ошибки загрузки."}
        else:
            comment["validation"] = {"type": "condition.any", "conditions": [
                {"type": "condition.not", "condition": status_is("uncertain")},
                {"type": "condition.required", "data": comment["data"]}],
                "hint": "Опишите, что вызвало сомнение."}
        items.append(comment)
        name = ("HW5: выделите взрослых бабочек рамками" if role == "bbox" else
                "HW5: проверьте рамки взрослых бабочек")
    template = {"view": {"type": "view.list", "items": items}, "plugins": [
        {"type": "plugin.toloka", "layout": {"kind": "scroll", "taskWidth": 1000}}]}
    components = set()

    def collect(value):
        if isinstance(value, dict):
            if (isinstance(value.get("type"), str) and "." in value["type"]
                    and not value["type"].startswith("data.")):
                components.add(value["type"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(template)
    lock = {component: "1.0.0" for component in components}
    lock.update({"core": "1.7.0", "view.list": "1.0.5", "condition.required": "1.1.5",
                 "plugin.toloka": "1.1.8"})
    return {"public_name": name, "public_description": name,
            "public_instructions": instructions_html, "private_comment": "HW5 " + role,
            "assignments_issuing_type": "AUTOMATED", "assignments_automerge_enabled": False,
            "task_spec": {"input_spec": inputs, "output_spec": outputs, "view_spec": {
                "type": "tb", "config": json.dumps(template, ensure_ascii=False), "lock": lock,
                "settings": {"showSkip": True, "showTimer": True, "showTitle": True,
                             "showFinish": True, "showSubmit": True, "showMessage": True,
                             "showFullscreen": True, "showInstructions": True, "showReward": True}}}}


def exam_suite(role, pool_id, controls=None):
    if role not in EXAM_QUESTIONS or not isinstance(pool_id, str) or not pool_id:
        raise ValueError("Exam role and pool_id are required")
    if controls is not None:
        if role != "review" or not isinstance(controls, list) or len(controls) != 10:
            raise ValueError("Visual review exam must contain exactly ten checked controls")
        tasks = []
        for control in controls:
            if (control.get("answer") not in {"yes", "no"}
                    or not isinstance(control.get("annotations"), list)
                    or not isinstance(control.get("image"), str) or not control["image"].startswith("https://")
                    or not isinstance(control.get("question"), str) or not control["question"].strip()):
                raise ValueError("Invalid visual review control")
            tasks.append({"input_values": {key: control[key] for key in ("question", "image", "annotations")},
                          "known_solutions": [{"output_values": {"answer": control["answer"]},
                                               "correctness_weight": 1}]})
        return {"pool_id": pool_id, "overlap": 20, "infinite_overlap": False, "tasks": tasks}
    return {"pool_id": pool_id, "overlap": 20, "infinite_overlap": False, "tasks": [
        {"input_values": {"question": question}, "known_solutions": [
            {"output_values": {"answer": answer}, "correctness_weight": 1}]}
        for question, answer in EXAM_QUESTIONS[role]]}


def task_suites(tasks, pool_id, per_page, overlap):
    if not isinstance(pool_id, str) or not pool_id or type(per_page) is not int or per_page < 1:
        raise ValueError("Pool and positive page size are required")
    if type(overlap) is not int or overlap < 1:
        raise ValueError("A finite positive overlap is required")
    return [{"pool_id": pool_id, "overlap": overlap, "infinite_overlap": False,
             "tasks": tasks[start:start + per_page]} for start in range(0, len(tasks), per_page)]


def build_pools(exam_project_id, bbox_project_id, review_project_id,
                bbox_exam_skill_id, review_exam_skill_id, bbox_author_skill_id, expires_at):
    ids = (exam_project_id, bbox_project_id, review_project_id, bbox_exam_skill_id,
           review_exam_skill_id, bbox_author_skill_id, expires_at)
    if not all(isinstance(value, str) and value for value in ids):
        raise ValueError("All project IDs, skill IDs and expiration must be supplied")
    if len({bbox_exam_skill_id, review_exam_skill_id, bbox_author_skill_id}) != 3:
        raise ValueError("Exam and participation skills must be distinct")
    result = {}
    for name, project, skill, reward, overlap in (
        ("bbox_exam", exam_project_id, bbox_exam_skill_id, .50, 20),
        ("review_exam", exam_project_id, review_exam_skill_id, .50, 20),
        ("bbox", bbox_project_id, bbox_exam_skill_id, .75, 1),
        ("review", review_project_id, review_exam_skill_id, .50, 2),
    ):
        exam = name.endswith("_exam")
        filters = [{"category": "profile", "key": "languages", "operator": "IN", "value": "RU"},
                   {"category": "skill", "key": skill, "operator": "EQ" if exam else "GTE",
                    "value": None if exam else 80}]
        if name in {"review", "review_exam"}:
            filters.append({"category": "skill", "key": bbox_author_skill_id,
                            "operator": "EQ", "value": None})
        configs = []
        if exam:
            configs = [{"collector_config": {"type": "GOLDEN_SET", "parameters": {"history_size": 10}},
                        "rules": [{"conditions": [{"key": "golden_set_answers_count", "operator": "GTE", "value": 10}],
                                   "action": {"type": "SET_SKILL_FROM_OUTPUT_FIELD", "parameters": {
                                       "skill_id": skill, "from_field": "golden_set_correct_answers_rate"}}}]}]
        result[name] = {"project_id": project, "private_name": "HW5 " + name,
                        "will_expire": expires_at, "reward_per_assignment": reward,
                        "assignment_max_duration_seconds": 600,
                        "auto_accept_solutions": True, "auto_close_after_complete_delay_seconds": 0,
                        "may_contain_adult_content": False,
                        "defaults": {"default_overlap_for_new_task_suites": overlap},
                        "filter": {"and": filters}, "quality_control": {"configs": configs}}
    return result


def maximum_cost(suites, reward, commission_percent, minimum_commission="0.001"):
    reward, commission_percent, minimum = map(Decimal, map(str, (
        reward, commission_percent, minimum_commission)))
    if not all(value.is_finite() and value >= 0 for value in (reward, commission_percent, minimum)):
        raise ValueError("Nonnegative finite prices are required")
    count = 0
    for suite in suites:
        if suite.get("infinite_overlap") is not False or type(suite.get("overlap")) is not int or suite["overlap"] < 1:
            raise ValueError("Every paid suite must have a finite positive overlap")
        count += suite["overlap"]
    fee = max(reward * commission_percent / 100, minimum) if reward else Decimal(0)
    return (count * (reward + fee)).quantize(Decimal("0.001"), rounding=ROUND_CEILING)


def validate_annotation(answer):
    if not isinstance(answer, dict) or answer.get("status") not in {value for value, _ in STATUS_OPTIONS}:
        raise ValueError("Unknown annotation status")
    boxes = answer.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    for box in boxes:
        if not isinstance(box, dict) or not all(
            type(box.get(key)) in (int, float) and math.isfinite(box[key])
            for key in ("left", "top", "width", "height")
        ):
            raise ValueError("Box coordinates must be finite numbers")
        left, top, width, height = (box[key] for key in ("left", "top", "width", "height"))
        if not (0 <= left < 1 and 0 <= top < 1 and width > 0 and height > 0
                and left + width <= 1 + 1e-9 and top + height <= 1 + 1e-9):
            raise ValueError("Invalid normalized box")
    if answer["status"] == "objects" and not boxes:
        raise ValueError("objects requires boxes")
    if answer["status"] in {"empty", "load_error"} and boxes:
        raise ValueError("empty/load_error cannot contain boxes")
    if answer["status"] == "uncertain" and (
        not isinstance(answer.get("comment"), str) or not answer["comment"].strip()
    ):
        raise ValueError("Uncertainty requires an explanation")


def normalize_annotations(annotations):
    """Convert confirmed Template Builder rectangles, rejecting other shapes."""
    if not isinstance(annotations, list):
        raise ValueError("Editor annotations must be a list")
    boxes = []
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("shape") != "rectangle":
            raise ValueError("Only rectangle annotations are supported")
        boxes.append({key: annotation.get(key) for key in ("left", "top", "width", "height")})
    validate_annotation({"status": "objects" if boxes else "empty", "boxes": boxes})
    return boxes


def review_decision(answer, reviews, assisted=False):
    validate_annotation(answer)
    if not answer.get("answer_id") or (not assisted and not answer.get("user_id")):
        raise ValueError("Answer identity and author are required")
    users, verdicts = set(), []
    for review in reviews:
        user = review.get("user_id")
        if (review.get("answer_id") != answer["answer_id"] or not user
                or (assisted and (not isinstance(user, str) or not user.strip()))):
            raise ValueError("Review has wrong answer_id or missing reviewer")
        if user == answer.get("user_id"):
            raise ValueError("Self-review is forbidden")
        if user in users:
            raise ValueError("Repeated reviewer is not independent")
        if review.get("verdict") not in {value for value, _ in VERDICT_OPTIONS}:
            raise ValueError("Unknown review verdict")
        users.add(user)
        verdicts.append(review["verdict"])
    if answer["status"] in {"load_error", "uncertain"}:
        return "manual"
    if len(verdicts) < 2:
        return "pending"
    if all(verdict == "accept" for verdict in verdicts):
        return "accept"
    if all(verdict == "reject" for verdict in verdicts):
        return "correct"
    return "manual"


def validate_golden(records, expected_ids):
    expected_ids = list(expected_ids)
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Manifest has duplicate IDs")
    seen = set()
    for record in records:
        photo_id = record.get("photo_id")
        if photo_id in seen or photo_id not in expected_ids:
            raise ValueError("Duplicate or unexpected golden ID")
        seen.add(photo_id)
        if record.get("error") is not None:
            raise ValueError("Unresolved error in golden set")
        validate_annotation({"status": "objects" if record.get("boxes") else "empty", "boxes": record.get("boxes")})
    if seen != set(expected_ids):
        raise ValueError("Golden set is incomplete")
