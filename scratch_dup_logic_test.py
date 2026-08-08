"""Unit-level check of the new tiered duplicate decision logic (isolated
from FAISS/HTTP, since it's pure conditional logic on already-known values)."""
from godhaar.config import DUPLICATE_THRESHOLD, DUPLICATE_HIGH_CONFIDENCE_THRESHOLD


def decide(score, body_match, muzzle_match):
    if score < DUPLICATE_THRESHOLD:
        return None  # loop breaks, never evaluated
    if score >= DUPLICATE_HIGH_CONFIDENCE_THRESHOLD:
        return True  # color cannot veto
    return muzzle_match  # ambiguous band: muzzle_color must corroborate


cases = [
    # (score, body_match, muzzle_match, expected, description)
    (0.95, False, False, True, "reported incident: high score, BOTH colors wrong -> must still reject"),
    (0.95, True, True, True, "high score, colors agree -> reject"),
    (0.85, True, True, True, "ambiguous band, both match -> reject"),
    (0.85, False, True, True, "ambiguous band, only muzzle matches -> reject (muzzle is trusted)"),
    (0.85, True, False, False, "ambiguous band, only body matches -> NOT rejected (body alone insufficient)"),
    (0.85, False, False, False, "ambiguous band, neither matches -> not a duplicate"),
    (0.79, True, True, None, "below base threshold -> never evaluated (loop breaks)"),
]

ok = True
for score, body, muzzle, expected, desc in cases:
    got = decide(score, body, muzzle)
    status = "PASS" if got == expected else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"[{status}] score={score} body={body} muzzle={muzzle} -> got={got} expected={expected} :: {desc}")

print("\nOVERALL:", "PASS" if ok else "FAIL")
