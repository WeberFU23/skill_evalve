"""
Agentic Memory System package.

Heavy modules are loaded lazily so optional dependencies such as ALFWorld do
not block importing lightweight modules like ``src.config`` or ``src.skill_tree``.
"""

from importlib import import_module


_EXPORTS = {
    "MemoryBank": ("src.memory_bank", "MemoryBank"),
    "OperationBank": ("src.operation_bank", "OperationBank"),
    "SkillTree": ("src.skill_tree", "SkillTree"),
    "SkillTreeSelector": ("src.skill_tree", "SkillTreeSelector"),
    "SkillNode": ("src.skill_tree", "SkillNode"),
    "SkillHardCase": ("src.skill_tree_evolution", "SkillHardCase"),
    "SkillHardCaseCollector": ("src.skill_tree_evolution", "SkillHardCaseCollector"),
    "SkillTreeEvolutionDesigner": ("src.skill_tree_evolution", "SkillTreeEvolutionDesigner"),
    "hard_case_from_selection": ("src.skill_tree_evolution", "hard_case_from_selection"),
    "PPOController": ("src.controller", "PPOController"),
    "Executor": ("src.executor", "Executor"),
    "Designer": ("src.designer", "Designer"),
    "BaseTrainer": ("src.trainer", "BaseTrainer"),
    "OfflineTrainer": ("src.trainer", "OfflineTrainer"),
    "get_trainer": ("src.trainer", "get_trainer"),
    "data_processing": ("src.data_processing", None),
    "eval": ("src.eval", None),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS.keys())
