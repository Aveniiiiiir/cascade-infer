"""__slots__ 一致性静态检查（审查 #1 的回归防线，无 torch 可跑）。

背景：ModelRunner/LLMEngine 曾在 __slots__ 中漏声明 draft/_so_* 等属性，
带 slots 的类无 __dict__ → 构造即 AttributeError → 整个引擎路径从未执行
过。本测试用 AST 扫描所有显式声明 __slots__ 的类，确保类体内任何
``self.<attr> = ...`` 赋值（含各方法）都在 slots 里——不导入模块，纯
源码分析，Windows 开发机即可拦截。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOTS = [Path(__file__).parents[2] / "cascade"]


def _iter_py_files():
    for root in _SRC_ROOTS:
        for p in root.rglob("*.py"):
            yield p


def _slots_of(cls_node: ast.ClassDef) -> set[str] | None:
    """类显式声明的 __slots__（元组字面量）；未声明返回 None。"""
    for node in cls_node.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            continue
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "__slots__":
                if isinstance(node.value, ast.Tuple):
                    names = set()
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.add(elt.value)
                        else:
                            return None  # 动态 slots（表达式）——跳过本类
                    return names
                return None
    return None


def _self_attrs(cls_node: ast.ClassDef) -> set[str]:
    """类体内所有 ``self.<attr> = ...``（含嵌套函数）的属性名。"""
    attrs: set[str] = set()

    class _V(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            if (
                isinstance(node.ctx, (ast.Store, ast.AugStore))
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                attrs.add(node.attr)
            self.generic_visit(node)

    _V().visit(cls_node)
    return attrs


@pytest.mark.parametrize("path", sorted(_iter_py_files()), ids=lambda p: p.name)
def test_slots_cover_all_self_assignments(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        slots = _slots_of(node)
        if slots is None:
            continue
        assigned = _self_attrs(node)
        missing = assigned - slots
        assert not missing, (
            f"{path.name}:{node.name} 声明了 __slots__ 但类内赋值了未声明的"
            f"属性 {sorted(missing)} —— 带slots类无 __dict__，构造即 "
            f"AttributeError（审查 #1 同类缺陷）"
        )
