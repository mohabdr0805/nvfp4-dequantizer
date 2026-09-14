"""
Telecharge le shard 1 du checkpoint NVFP4 : le vrai fichier d'entree.

    python fetch_shard.py

4,98 Go dans bench/ (ignore par git). 1026 tenseurs, dont 224 en NVFP4.
C'est la seule facon de prouver que le streaming tient ses promesses : 5 Go
en entree ne tiennent pas dans un tampon.

Reprend ou il s'est arrete si on le relance apres une coupure.
"""

import os
import sys
import urllib.request

REPO = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
FICHIER = "model-00001-of-00002.safetensors"
DOSSIER = "bench"


def main():
    url = f"https://huggingface.co/{REPO}/resolve/main/{FICHIER}"
    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, FICHIER)

    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers["Content-Length"])

    deja = os.path.getsize(chemin) if os.path.exists(chemin) else 0
    if deja == total:
        print(f"deja complet : {chemin} ({total / 1e9:.2f} Go)")
        return
    if deja:
        print(f"reprise a {deja / 1e9:.2f} Go sur {total / 1e9:.2f}")

    req = urllib.request.Request(url)
    if deja:
        req.add_header("Range", f"bytes={deja}-")

    with urllib.request.urlopen(req) as rep, open(chemin, "ab") as f:
        lu = deja
        while True:
            morceau = rep.read(4 << 20)
            if not morceau:
                break
            f.write(morceau)
            lu += len(morceau)
            pct = 100 * lu / total
            sys.stdout.write(f"\r  {lu / 1e9:5.2f} / {total / 1e9:.2f} Go  {pct:5.1f} %")
            sys.stdout.flush()

    print(f"\n{chemin} : {os.path.getsize(chemin) / 1e9:.2f} Go")


if __name__ == "__main__":
    main()
