"""Check empirically whether Earth Search pixels already have the BOA offset removed.

Earth Search lists some acquisitions twice: the original processing (e.g. baseline
03.00, no offset) and an ESA reprocessing (baseline 05.00, which adds +1000 to every DN).
The reprocessed item declares raster:bands offset -0.1 and earthsearch:boa_offset_applied.

If the +1000 were still in the reprocessed pixels, its DNs would be ~1000 higher than
the original's for the same ground. If they match, the offset was already removed and
applying -0.1 again would double-subtract.

Part 2 shows what double-subtracting would do to the README's worked example scene.

    python scripts/verify_boa_offset.py
"""

from __future__ import annotations

import numpy as np
from pystac_client import Client

from orbitwatch.eo.indices import compute_index, dn_to_reflectance, scl_valid_mask
from orbitwatch.eo.raster import read_aligned_bands
from orbitwatch.eo.stac import COLLECTION, EARTH_SEARCH_URL, scene_from_item

PAIR = ["S2A_10SGD_20210615_0_L2A", "S2A_10SGD_20210615_1_L2A"]
AREAS = {
    "Pacific Ocean off Point Arguello": [-120.72, 34.55, -120.70, 34.57],
    "Lompoc Valley fields": [-120.50, 34.63, -120.48, 34.65],
}
EXAMPLE_SCENE = "S2C_10SGD_20260420_0_L2A"
EXAMPLE_AOI = [-120.53, 34.635, -120.49, 34.665]


def main() -> None:
    client = Client.open(EARTH_SEARCH_URL)
    items = {
        it.id: it.to_dict() for it in client.search(collections=[COLLECTION], ids=PAIR).items()
    }
    medians = {}
    for scene_id in PAIR:
        scene = scene_from_item(items[scene_id])
        print(
            f"{scene_id}: baseline {scene.processing_baseline}, "
            f"boa_offset_applied={scene.boa_offset_applied}, "
            f"declared red offset {scene.assets['red'].declared_offset}"
        )
        for area, bbox in AREAS.items():
            raw, _ = read_aligned_bands({"red": scene.assets["red"].href}, bbox, 10.0)
            dn = raw["red"][raw["red"] > 0]
            medians[(scene_id, area)] = float(np.median(dn))
            print(f"    {area:<34} red DN median {np.median(dn):7.0f}  (n={dn.size})")
    diffs = []
    for area in AREAS:
        diffs.append(medians[(PAIR[1], area)] - medians[(PAIR[0], area)])
        print(f"{area}: reprocessed minus original median DN = {diffs[-1]:+.0f}")
    print(
        "Offset still in pixels: apply -0.1"
        if min(diffs) > 500
        else "DNs match: offset already removed; do not apply -0.1 again"
    )

    impact(client)


def impact(client) -> None:
    """NDVI of the worked-example scene with the offset skipped vs. applied again."""
    item = next(client.search(collections=[COLLECTION], ids=[EXAMPLE_SCENE]).items())
    scene = scene_from_item(item.to_dict())
    hrefs = {k: scene.assets[k].href for k in ("nir", "red", "scl")}
    raw, _ = read_aligned_bands(hrefs, EXAMPLE_AOI, 10.0)
    valid = scl_valid_mask(raw["scl"])
    print(f"\n{EXAMPLE_SCENE} over {EXAMPLE_AOI}: red DN median {np.median(raw['red']):.0f}")
    for label, offset in [("offset skipped (orbitwatch)", 0.0), ("declared -0.1 applied", -0.1)]:
        refl = {k: dn_to_reflectance(raw[k], 0.0001, offset, 0) for k in ("nir", "red")}
        ndvi = compute_index("ndvi", refl, valid)
        print(
            f"    {label:<28} mean NDVI {np.nanmean(ndvi):.4f}, valid pixels "
            f"{np.isfinite(ndvi).mean() * 100:.2f}%, red reflectance < 0 in "
            f"{np.mean(refl['red'] < 0) * 100:.2f}% of pixels"
        )


if __name__ == "__main__":
    main()
