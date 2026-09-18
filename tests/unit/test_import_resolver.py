from repochat.graph.import_resolver import resolve_javascript_import, resolve_python_import


def test_resolves_relative_python_import_same_package():
    path_index = {"nanochat/common.py": "FILE:common", "nanochat/trainer.py": "FILE:trainer"}
    result = resolve_python_import("from .common import setup", "nanochat/trainer.py", path_index)
    assert result == "FILE:common"


def test_resolves_relative_python_import_parent_package():
    path_index = {"pkg/utils.py": "FILE:utils", "pkg/sub/mod.py": "FILE:mod"}
    result = resolve_python_import("from ..utils import helper", "pkg/sub/mod.py", path_index)
    assert result == "FILE:utils"


def test_resolves_absolute_python_import_within_repo():
    path_index = {"nanochat/common.py": "FILE:common"}
    result = resolve_python_import("from nanochat.common import setup", "scripts/train.py", path_index)
    assert result == "FILE:common"


def test_resolves_bare_relative_import_to_package_init():
    # "from . import x" (no module name) -- the current package itself is
    # the target, represented by its __init__.py.
    path_index = {"nanochat/__init__.py": "FILE:init"}
    result = resolve_python_import("from . import nanochat", "nanochat/sub.py", path_index)
    assert result == "FILE:init"


def test_external_python_import_is_unresolved():
    path_index = {"nanochat/common.py": "FILE:common"}
    result = resolve_python_import("import torch", "nanochat/common.py", path_index)
    assert result is None


def test_resolves_relative_javascript_import_with_extension_added():
    path_index = {"source/utils.js": "FILE:utils", "source/index.js": "FILE:index"}
    result = resolve_javascript_import("import { helper } from './utils'", "source/index.js", path_index)
    assert result == "FILE:utils"


def test_resolves_relative_javascript_require():
    path_index = {"lib/config.js": "FILE:config"}
    result = resolve_javascript_import("const config = require('../lib/config')", "src/app.js", path_index)
    assert result == "FILE:config"


def test_resolves_javascript_directory_index():
    path_index = {"components/button/index.js": "FILE:button_index"}
    result = resolve_javascript_import("import Button from './button'", "components/app.js", path_index)
    assert result == "FILE:button_index"


def test_external_javascript_package_is_unresolved():
    path_index = {"source/index.js": "FILE:index"}
    result = resolve_javascript_import("import chalk from 'chalk'", "source/index.js", path_index)
    assert result is None
