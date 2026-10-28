"""Unit tests for the evaluation pass/fail criteria."""

from scripts.evaluate import _evaluation_passed


def _successful_record() -> dict:
    return {
        "ingestion_success": True,
        "plan_created": True,
        "workflow_status": "completed",
        "execution_success": True,
        "reviewer_accepted": True,
        "report_generated": True,
        "is_fallback": False,
        "fallback_used": False,
        "live_review_success": True,
    }


def test_live_evaluation_passes_only_with_live_review_and_no_fallback():
    record = _successful_record()
    assert _evaluation_passed(record, demo_mode=False)


def test_live_evaluation_fails_if_reviewer_used_fallback():
    record = _successful_record()
    record["live_review_success"] = False
    record["is_fallback"] = True
    record["fallback_used"] = True
    assert not _evaluation_passed(record, demo_mode=False)


def test_live_evaluation_fails_if_any_stage_fell_back_even_when_review_was_live():
    record = _successful_record()
    record["fallback_used"] = True
    assert not _evaluation_passed(record, demo_mode=False)


def test_demo_evaluation_can_pass_without_live_review():
    record = _successful_record()
    record["live_review_success"] = False
    assert _evaluation_passed(record, demo_mode=True)


def test_report_alone_does_not_count_as_pass():
    record = _successful_record()
    record["execution_success"] = False
    assert not _evaluation_passed(record, demo_mode=True)
