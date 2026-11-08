"""Strict grading of model answers."""

import json


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonstandard JSON constant: " + value)


def score_answer(content, expected, finish_reason="stop"):
    """Distinguish strict JSON, answer schema, and solution correctness.

    Duplicate keys and nonstandard numeric constants are invalid JSON here.
    A valid but wrong answer has no validation error. Truncation invalidates
    the format even when the returned text happens to contain a full object.
    """
    result = {"json_valid": False, "format_valid": False,
              "correct": False, "validation_error": None}
    if not isinstance(content, str):
        result["validation_error"] = "Content must be a JSON string"
        return result
    try:
        answer = json.loads(content, object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        result["validation_error"] = "Invalid JSON: " + str(error)
        return result
    result["json_valid"] = True
    if finish_reason == "length":
        result["validation_error"] = "Response truncated: finish_reason=length"
    elif finish_reason != "stop":
        result["validation_error"] = "Normal completion was not confirmed: finish_reason != stop"
    elif not isinstance(answer, dict) or set(answer) != {"answer"}:
        result["validation_error"] = "Expected a single object with only the answer key"
    elif not isinstance(answer["answer"], str) or answer["answer"] not in ("A", "B", "C", "D"):
        result["validation_error"] = "Answer must be one of A, B, C, D"
    else:
        result["format_valid"] = True
        result["correct"] = answer["answer"] == expected
    return result
