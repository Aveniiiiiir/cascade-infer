"""根 conftest：把项目根加入 sys.path——裸 `pytest`（非 `python -m pytest`）
也能导入 cascade 包（未 pip install 的开发环境）。

集群正式环境用 `pip install -e . --no-deps` 安装后本文件无副作用。
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
