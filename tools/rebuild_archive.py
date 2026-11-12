#!/usr/bin/env python3
"""Rebuild the public archive's chart-read sequence from git history."""
import json, subprocess, sys
from pathlib import Path

REPO = Path.home() / "makefun/archive"
OUT = Path("/tmp/rbxv3/archive_reads.ndjson")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


commits = [l.split()[0] for l in git("log", "--reverse", "--format=%H %ci", "--", "data/discover/latest.json").splitlines() if l.strip()]
rows = []
for sha in commits:
    blob = git("show", f"{sha}:data/discover/latest.json")
    if not blob.strip():
        continue
    doc = json.loads(blob)
    rank = {}
    for cand in doc.get("candidates") or []:
        for src in cand.get("sources") or []:
            if src.get("source") == "explore:up-and-coming" and src.get("rank"):
                rank[int(cand["universe_id"])] = int(src["rank"])
    if not rank:
        continue
    rows.append({"at": doc.get("generated_at"), "rank": rank, "commit": sha[:10]})
rows.sort(key=lambda r: r["at"])
with open(OUT, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("reads", len(rows), rows[0]["at"], "->", rows[-1]["at"])
for r in rows:
    top = sorted(r["rank"].items(), key=lambda kv: kv[1])[:3]
    print(r["at"], len(r["rank"]), top)
