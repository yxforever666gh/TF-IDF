from __future__ import annotations

from datetime import date, timedelta

from unified_tfidf.cleaning import content_status, normalize_text, parse_date
from unified_tfidf.privacy import contains_pii, sanitize_text


def test_normalization_and_invalid_content_rules() -> None:
    assert normalize_text("  <b>ＡＢＣ</b>\n  测试  ") == "ABC 测试"
    assert content_status("") == "空文本"
    assert content_status("该用户未填写评价内容") == "平台模板"
    assert content_status("很好") == "低信息"


def test_sensitive_patterns_are_replaced() -> None:
    source = "电话13812345678，邮箱demo@example.com，身份证440101199001011234，订单号:JD123456789"
    sanitized, replacements = sanitize_text(source)
    assert replacements == 4
    assert "[已脱敏]" in sanitized
    assert not contains_pii(sanitized)


def test_future_dates_are_flagged_not_removed() -> None:
    future = (date.today() + timedelta(days=10)).isoformat()
    parsed, status = parse_date(future)
    assert parsed == future
    assert status == "日期异常"

