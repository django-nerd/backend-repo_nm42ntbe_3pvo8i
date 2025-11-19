import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Chord Progression API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Music theory utilities
# ------------------------
NOTE_ORDER_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_INTERVALS = [2, 2, 1, 2, 2, 2, 1]  # W W H W W W H
NAT_MINOR_INTERVALS = [2, 1, 2, 2, 1, 2, 2]  # W H W W H W W

MODE_INTERVALS = {
    "major": MAJOR_INTERVALS,
    "ionian": MAJOR_INTERVALS,
    "natural_minor": NAT_MINOR_INTERVALS,
    "minor": NAT_MINOR_INTERVALS,
    "aeolian": NAT_MINOR_INTERVALS,
}

DEGREE_ROMAN_MAJOR = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
DEGREE_ROMAN_MINOR = ["i", "ii°", "III", "iv", "v", "VI", "VII"]

COMMON_PROGRESSIONS_MAJOR = [
    [1, 5, 6, 4],
    [1, 4, 5, 1],
    [2, 5, 1, 6],
    [6, 4, 1, 5],
]
COMMON_PROGRESSIONS_MINOR = [
    [1, 7, 6, 7],
    [6, 7, 1, 5],
    [1, 6, 3, 7],
    [1, 4, 7, 3],
]


def note_index(note: str) -> int:
    name = note.strip().upper()
    # Support b (flats) by converting to enharmonic sharps
    flat_map = {
        "DB": "C#",
        "EB": "D#",
        "GB": "F#",
        "AB": "G#",
        "BB": "A#",
    }
    if name.endswith("B") and len(name) == 2 and name not in ["AB", "BB"]:
        name = flat_map.get(name, name)
    name = name.replace("B", "b")  # keep Ab/Bb mapping
    enharmonic = {
        "CB": "B", "B#": "C", "E#": "F", "FB": "E",
        "AB": "G#", "BB": "A#", "DB": "C#", "EB": "D#", "GB": "F#"
    }
    base = enharmonic.get(name, name)
    if base not in NOTE_ORDER_SHARPS:
        # try flats like Ab, Bb
        if len(base) == 2 and base[1] == 'b':
            # convert: A b => A (9) - 1 => 8 => G#
            letter = base[0]
            idx = {n[0]: i for i, n in enumerate(NOTE_ORDER_SHARPS) if len(n) == 1}.get(letter, None)
            if idx is None:
                raise ValueError(f"Invalid note: {note}")
            return (idx - 1) % 12
        raise ValueError(f"Invalid note: {note}")
    return NOTE_ORDER_SHARPS.index(base)


def build_scale(root: str, mode: str) -> List[int]:
    intervals = MODE_INTERVALS.get(mode.lower())
    if not intervals:
        raise ValueError("Unsupported mode. Use 'major' or 'minor'.")
    start = note_index(root)
    degrees = [start]
    for step in intervals[:-1]:  # 6 steps to get 7-note scale
        degrees.append((degrees[-1] + step) % 12)
    return degrees


def stack_thirds(scale_pcs: List[int], degree: int, num_notes: int) -> List[int]:
    # degree: 1-7; stack 3rds within the diatonic scale
    pcs = []
    for k in range(num_notes):
        pcs.append(scale_pcs[(degree - 1 + 2 * k) % 7])
    return pcs


def name_for_pc(pc: int) -> str:
    return NOTE_ORDER_SHARPS[pc]


def midi_for_pc(pc: int, octave: int = 4) -> int:
    # C4 = 60
    base_c4 = 60
    c_pc = 0  # C
    semitone_from_c = (pc - c_pc) % 12
    return base_c4 + semitone_from_c + (octave - 4) * 12


def ensure_spread_midi(pcs: List[int], base_octave: int = 4) -> List[int]:
    # Spread chord notes upwards to avoid duplicates; ensure ascending order
    used = set()
    midis: List[int] = []
    cur_oct = base_octave
    last_midi = None
    for pc in pcs:
        m = midi_for_pc(pc, cur_oct)
        if last_midi is not None and m <= last_midi:
            while m <= last_midi:
                cur_oct += 1
                m = midi_for_pc(pc, cur_oct)
        midis.append(m)
        last_midi = m
    return midis


def frequencies_from_midi(midis: List[int]) -> List[float]:
    return [440.0 * (2 ** ((m - 69) / 12)) for m in midis]


# ------------------------
# Models
# ------------------------
class ScaleRequest(BaseModel):
    root: str = Field(..., description="Root note, e.g., C, C#, D, Eb")
    mode: str = Field("major", description="'major' or 'minor'")


class ProgressionRequest(BaseModel):
    root: str
    mode: str = Field("major")
    length: int = Field(4, ge=1, le=12)
    chord_types: List[str] = Field(default_factory=lambda: ["triad"])  # triad, seventh, sus2, sus4, power, add9
    degrees: Optional[List[int]] = Field(None, description="Optional explicit degrees (1-7)")


# ------------------------
# Routes
# ------------------------
@app.get("/")
def read_root():
    return {"message": "Chord Progression API ready"}


@app.post("/api/scale")
def api_scale(req: ScaleRequest) -> Dict[str, Any]:
    try:
        pcs = build_scale(req.root, req.mode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    names = [name_for_pc(pc) for pc in pcs]
    romans = DEGREE_ROMAN_MAJOR if req.mode.lower() in ("major", "ionian") else DEGREE_ROMAN_MINOR
    return {"root": req.root, "mode": req.mode, "pitch_classes": pcs, "notes": names, "degrees": romans}


def build_chord_for_degree(scale_pcs: List[int], degree: int, ctype: str) -> Dict[str, Any]:
    ctype = ctype.lower()
    pcs: List[int]
    if ctype == "triad":
        pcs = stack_thirds(scale_pcs, degree, 3)
    elif ctype == "seventh":
        pcs = stack_thirds(scale_pcs, degree, 4)
    elif ctype == "sus2":
        root_pc = scale_pcs[degree - 1]
        pcs = [root_pc, (root_pc + 2) % 12, (root_pc + 7) % 12]
    elif ctype == "sus4":
        root_pc = scale_pcs[degree - 1]
        pcs = [root_pc, (root_pc + 5) % 12, (root_pc + 7) % 12]
    elif ctype == "power":
        root_pc = scale_pcs[degree - 1]
        pcs = [root_pc, (root_pc + 7) % 12]
    elif ctype == "add9":
        triad = stack_thirds(scale_pcs, degree, 3)
        pcs = triad + [ (triad[0] + 14) % 12 ]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported chord type: {ctype}")

    midis = ensure_spread_midi(pcs, base_octave=4)
    freqs = frequencies_from_midi(midis)
    name = f"{name_for_pc(scale_pcs[degree-1])}{ctype if ctype != 'triad' else ''}"
    return {
        "degree": degree,
        "type": ctype,
        "name": name,
        "pcs": pcs,
        "notes": [name_for_pc(pc) for pc in pcs],
        "midi": midis,
        "frequencies": freqs,
    }


@app.post("/api/generate-progression")
def generate_progression(req: ProgressionRequest) -> Dict[str, Any]:
    try:
        scale_pcs = build_scale(req.root, req.mode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Choose degrees
    degrees: List[int]
    if req.degrees:
        degrees = [((d - 1) % 7) + 1 for d in req.degrees]
    else:
        import random
        template = random.choice(COMMON_PROGRESSIONS_MAJOR if req.mode.lower() in ("major", "ionian") else COMMON_PROGRESSIONS_MINOR)
        # Extend or trim to requested length
        while len(template) < req.length:
            template += template  # repeat
        degrees = template[:req.length]

    # Build a chord for each step using the first available chord type preference
    chords: List[Dict[str, Any]] = []
    types_cycle = req.chord_types if req.chord_types else ["triad"]
    for i, deg in enumerate(degrees):
        ctype = types_cycle[i % len(types_cycle)]
        chord = build_chord_for_degree(scale_pcs, deg, ctype)
        chords.append(chord)

    romans = DEGREE_ROMAN_MAJOR if req.mode.lower() in ("major", "ionian") else DEGREE_ROMAN_MINOR
    return {
        "root": req.root,
        "mode": req.mode,
        "degrees": degrees,
        "roman": [romans[d-1] for d in degrees],
        "chords": chords,
        "scale": [name_for_pc(pc) for pc in scale_pcs],
    }


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used (no persistence required)",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    # Environment variables presence (for info)
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
