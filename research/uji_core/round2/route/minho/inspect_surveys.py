"""Describe registered surveys from UM DSI lite without decoding signal values."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def describe(archive):
    with archive.open("rb") as source:
        md5 = hashlib.file_digest(source, "md5").hexdigest()
    if md5 != "4c15884cb7edcd2c602ec5d567ddd8a8":
        raise ValueError("Expected the published 6646008 lite release")

    visits = []
    sites_by_date = []
    with ZipFile(archive) as zipped:
        files = sorted(name for name in zipped.namelist()
                       if name.startswith("UM_DSI_DB_v1.0.0_lite/data/site_surveys/")
                       and name.endswith("/coordinates.csv"))
        for name in files:
            raw = zipped.read(name)
            points = [tuple(map(float, row)) for row in csv.reader(io.StringIO(raw.decode()))]
            positions = set(points)
            visits.append({"date": PurePosixPath(name).parent.name, "rows": len(points),
                           "unique_coordinates": len(positions), "z": sorted({p[2] for p in points}),
                           "coordinate_sha256": hashlib.sha256(raw).hexdigest()})
            sites_by_date.append(positions)
    return {"doi": "10.5281/zenodo.6646008", "release_md5": md5,
            "format": "survey coordinates only; no RSS values or monitor rows decoded",
            "visits": visits, "total_rows": sum(v["rows"] for v in visits),
            "exact_union": len(set.union(*sites_by_date)),
            "exact_intersection": len(set.intersection(*sites_by_date)),
            "consecutive_disjoint_visit_pairs": [
                {"earlier": a["date"], "later": b["date"], "common_xyz": len(x & y)}
                for a, b, x, y in zip(visits[::2], visits[1::2], sites_by_date[::2], sites_by_date[1::2])
            ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(describe(args.archive), indent=2))
