"""Offline tests for benchmark answer generation helpers."""

from __future__ import annotations

from judge.generate_answers import build_answer_system, strip_yaml_frontmatter


def test_strip_yaml_frontmatter_removes_block() -> None:
    raw = "---\nname: x\n---\n\n# Hello\nbody"
    assert strip_yaml_frontmatter(raw).startswith("# Hello")


def test_strip_yaml_frontmatter_no_frontmatter() -> None:
    raw = "# Title\nno yaml"
    assert strip_yaml_frontmatter(raw) == raw


def test_build_answer_system_without_skill() -> None:
    s = build_answer_system(None)
    low = s.lower()
    assert "medical ai" in low
    assert "referenced" in low


def test_build_answer_system_with_skill_body() -> None:
    s = build_answer_system("---\na: 1\n---\n\nSKILL LINE")
    assert "SKILL LINE" in s
    assert "a: 1" not in s
