"""Contre-epreuve : lit un gros fichier avec N fils, hors de Zig.

    python outils/lecture.py <fichier> <fils> [alterne]

Sert a savoir si un plafond de lecture vient du peripherique ou du programme.
Chaque fil ouvre SON descripteur -- c'est la difference qui a revele le bogue du
19/09, ou les ouvriers Zig partageaient un seul `std.Io.File` et ne montaient pas.

Par defaut chaque fil lit une tranche contigue ; avec `alterne`, ils se partagent
les morceaux en tourniquet, ce qui imite la distribution dynamique du programme.

Vider le cache du systeme avant chaque mesure : `zig run outils/videcache.zig`.
Sinon on mesure une recopie memoire, pas le disque.
"""
import os
import sys
import threading
import time

CHUNK = 4 * 1024 * 1024


def contigu(chemin, debut, fin):
    buf = bytearray(CHUNK)
    with open(chemin, "rb", buffering=0) as f:
        f.seek(debut)
        reste = fin - debut
        while reste > 0:
            n = f.readinto(memoryview(buf)[: min(CHUNK, reste)])
            if not n:
                break
            reste -= n


def alterne(chemin, moi, fils, n_chunks, taille):
    buf = bytearray(CHUNK)
    with open(chemin, "rb", buffering=0) as f:
        k = moi
        while k < n_chunks:
            f.seek(k * CHUNK)
            f.readinto(memoryview(buf)[: min(CHUNK, taille - k * CHUNK)])
            k += fils


def main():
    chemin, fils = sys.argv[1], int(sys.argv[2])
    tourniquet = len(sys.argv) > 3 and sys.argv[3] == "alterne"
    taille = os.path.getsize(chemin)

    if tourniquet:
        n = (taille + CHUNK - 1) // CHUNK
        ts = [threading.Thread(target=alterne, args=(chemin, i, fils, n, taille))
              for i in range(fils)]
    else:
        pas = taille // fils
        bornes = [(i * pas, taille if i == fils - 1 else (i + 1) * pas)
                  for i in range(fils)]
        ts = [threading.Thread(target=contigu, args=(chemin, a, b)) for a, b in bornes]

    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    s = time.perf_counter() - t0

    mode = "alterne" if tourniquet else "contigu"
    print(f"{fils:2d} fil(s) {mode} : {s:5.2f} s   {taille / 1e9 / s:5.2f} Go/s")


main()
