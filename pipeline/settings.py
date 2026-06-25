"""User-facing settings with spec-default values (spec §8). All taste controls
live here so the UI, single-track and album paths share one schema."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Settings:
    # --- master mode ---
    mode: str = "genre"          # "genre" | "my_reference" | "auto"

    # --- taste sliders ---
    vocal_level_db: float = 0.0  # ±6 dB around the balanced default (beat +1 LU)
    width: float = 0.5           # 0 (subtle) .. 1 (wide)
    warmth: float = 0.4          # 0 (off) .. 1 (strong)
    loudness_target: float = -9.0  # LUFS, slider range -12 .. -7
    glue_reverb: bool = False    # optional very-short plate, OFF by default
    offset_ms: float = 0.0       # manual latency nudge on the vocal

    # --- toggles (sensible pro defaults) ---
    deesser: bool = True
    ducking: bool = True
    mixbus_glue: bool = True

    # --- fixed engineering constants (spec §8) ---
    vocal_hpf_hz: float = 90.0
    vocal_balance_offset_lu: float = 1.0   # vocal ≈ beat +1 LU
    premaster_headroom_db: float = -6.0
    mono_below_hz: float = 120.0
    width_above_hz: float = 300.0
    true_peak_ceiling_db: float = -1.0
    oversample: int = 4

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        d = dict(d or {})
        fields = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in fields}
        s = cls(**clean)
        s.clamp()
        return s

    def clamp(self) -> None:
        self.vocal_level_db = float(max(-6.0, min(6.0, self.vocal_level_db)))
        self.width = float(max(0.0, min(1.0, self.width)))
        self.warmth = float(max(0.0, min(1.0, self.warmth)))
        self.loudness_target = float(max(-12.0, min(-7.0, self.loudness_target)))
        self.mode = self.mode if self.mode in ("genre", "my_reference", "auto") else "genre"

    def to_dict(self) -> dict:
        return asdict(self)
