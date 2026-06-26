"""Genre "Sounds" — voiced mastering presets for the Genre selector.

Each Sound is a target tonal curve plus a set of signature-chain modifiers,
voiced from the artists the user associates with that genre. We can't bundle the
artists' actual records (copyright), so a Sound is a *voicing* that gets a mix
into that ballpark; "My reference" mode still exists for matching a specific song.

Two house characters sit under EVERY Sound (see master.signature_chain):
  - Kanye (primary):     soulful low-mid body, even-harmonic warmth, full & weighty.
  - untiljapan (light):  a smooth-air + width baseline so nothing sounds closed-in.

Sounds (artists they were voiced from):
  - trap     — Young Thug, Future, Travis Scott  (dark, 808-heavy, saturated, wide, loud)
  - boombap  — Kanye, JID, Joey Bada$$, Griselda, Larry June  (warm, mid-forward, punchy, dynamic)
  - melodic  — untiljapan, Lancey Foux, Smino, Isaiah Rashad  (lush, smooth, wide, silky top)
"""
from __future__ import annotations

from dataclasses import dataclass


# Relative target tilts (dB) — only the *shape* matters; matching_eq mean-
# normalises each curve on a mid reference band before use.
TRAP_TARGET = [
    (30, 1.8), (50, 2.4), (80, 2.0), (120, 1.2), (200, 0.3), (300, -0.3),
    (500, -0.4), (800, -0.2), (1000, 0.0), (2000, -0.2), (3500, -0.6),
    (5000, -0.9), (7000, -1.0), (9000, -0.7), (12000, -0.3), (16000, -0.2),
    (20000, -0.9),
]
BOOMBAP_TARGET = [
    (30, 0.4), (50, 1.0), (80, 1.3), (120, 1.2), (200, 0.9), (300, 0.5),
    (500, 0.3), (800, 0.2), (1000, 0.0), (2000, 0.0), (3500, -0.2),
    (5000, -0.5), (7000, -0.6), (9000, -0.6), (12000, -0.6), (16000, -0.6),
    (20000, -1.0),
]
MELODIC_TARGET = [
    (30, 0.9), (50, 1.5), (80, 1.3), (120, 0.8), (200, -0.1), (300, -0.5),
    (500, -0.4), (800, -0.2), (1000, 0.0), (2000, 0.2), (3500, -0.1),
    (5000, -0.2), (7000, -0.2), (9000, 0.1), (12000, 0.5), (16000, 0.6),
    (20000, -0.4),
]


@dataclass(frozen=True)
class SoundProfile:
    key: str
    label: str
    target: list                 # tonal target anchors
    low_weight_db: float         # extra low-shelf weight (808/sub heft)
    sat_mult: float              # saturation drive multiplier (grit/warmth)
    width_mult: float            # stereo-width multiplier
    presence_db: float           # 2.8 kHz upper-mid "cut"
    air_db: float                # 11 kHz air shelf
    dynamic: float               # 1.0 = dense; >1 keeps more dynamics (lighter glue)


SOUNDS: dict[str, SoundProfile] = {
    # Dark, 808-heavy, saturated, wide, loud (Young Thug / Future / Travis).
    "trap": SoundProfile(
        "trap", "Trap", TRAP_TARGET,
        low_weight_db=2.6, sat_mult=1.35, width_mult=1.15,
        presence_db=0.7, air_db=1.0, dynamic=1.0,
    ),
    # Warm, mid-forward, punchy, sample-dusty — more dynamic, narrower, less air
    # (Kanye / JID / Joey / Griselda / Larry June).
    "boombap": SoundProfile(
        "boombap", "Boom-bap", BOOMBAP_TARGET,
        low_weight_db=0.4, sat_mult=1.2, width_mult=0.65,
        presence_db=2.3, air_db=0.4, dynamic=1.35,
    ),
    # Lush, smooth, widest, silky/airy controlled top — singer-rapper sound
    # (untiljapan / Lancey Foux / Smino / Isaiah Rashad).
    "melodic": SoundProfile(
        "melodic", "Melodic / R&B", MELODIC_TARGET,
        low_weight_db=1.0, sat_mult=0.7, width_mult=1.35,
        presence_db=1.2, air_db=2.8, dynamic=1.05,
    ),
}

DEFAULT_SOUND = "trap"
SOUND_KEYS = tuple(SOUNDS.keys())


def get_sound(name: str | None) -> SoundProfile:
    return SOUNDS.get(name or "", SOUNDS[DEFAULT_SOUND])
