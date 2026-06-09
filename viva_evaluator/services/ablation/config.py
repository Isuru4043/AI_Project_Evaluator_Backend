"""
Ablation flags — toggles individual contributions on/off for experiments.
"""

from dataclasses import dataclass


@dataclass
class AblationFlags:
    """
    Each flag, when True, DISABLES that contribution. Defaults are False
    (= full system enabled = baseline).
    """
    disable_anchoring:        bool = False
    disable_tier1_validation: bool = False
    disable_critic:           bool = False
    disable_kg:               bool = False
    disable_section_aware:    bool = False

    def label(self) -> str:
        """Short label for tables: 'full' if all defaults, else flag list."""
        flags = [name for name, val in self.__dict__.items() if val]
        return 'full_system' if not flags else '|'.join(
            f.replace('disable_', 'no_') for f in flags
        )

    @classmethod
    def from_dict(cls, data) -> 'AblationFlags':
        if not isinstance(data, dict):
            return cls()
        return cls(
            disable_anchoring        = bool(data.get('disable_anchoring', False)),
            disable_tier1_validation = bool(data.get('disable_tier1_validation', False)),
            disable_critic           = bool(data.get('disable_critic', False)),
            disable_kg               = bool(data.get('disable_kg', False)),
            disable_section_aware    = bool(data.get('disable_section_aware', False)),
        )
