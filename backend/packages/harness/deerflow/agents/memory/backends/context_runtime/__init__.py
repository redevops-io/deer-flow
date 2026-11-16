"""context-runtime memory backend -- HTTP client against context-runtime's control-plane
``/memory/*`` routes (bi-temporal fact storage).

Drop-in contract: folder name == backend name == ``manager_class: context_runtime``.
"""

from .context_runtime_manager import ContextRuntimeManager

#: Discovered by the factory's ``_scan_backends`` under the folder name ``context_runtime``.
MANAGER_CLASS = ContextRuntimeManager
