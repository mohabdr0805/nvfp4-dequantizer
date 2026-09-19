"""Verifie la sortie du dequantiseur contre l'entree, tensor par tensor.

    python verif.py <entree.safetensors> <sortie> [reference/]

Controles :
  1. taille exacte du fichier de sortie
  2. en-tete : nombre de tenseurs, offsets contigus
  3. la ligne 0 de q_proj contre le fichier de reference, bit a bit
  4. pour CHAQUE tenseur : debut et fin recalcules independamment en Python
  5. pour les gros tenseurs : les bords de chaque morceau de 4 Mo
"""

import json
import math
import struct
import sys

import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                dtype=np.float32)


def table_e4m3():
    t = np.empty(256, dtype=np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            t[b] = np.nan
        elif e == 0:
            t[b] = (m / 8) * 2.0 ** -6
        else:
            t[b] = (1 + m / 8) * 2.0 ** (e - 7)
        if s:
            t[b] = -t[b]
    return t


E4M3 = table_e4m3()


def entete(f):
    f.seek(0)
    n = struct.unpack("<Q", f.read(8))[0]
    return 8 + n, json.loads(f.read(n))


def lire(f, debut, longueur):
    f.seek(debut)
    b = f.read(longueur)
    assert len(b) == longueur, f"lecture courte a {debut}: {len(b)}/{longueur}"
    return b


CHUNK = 4 * 1024 * 1024


def main():
    chemin_in, chemin_out = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) > 3 else "reference/"

    fi = open(chemin_in, "rb")
    fo = open(chemin_out, "rb")
    base_in, h_in = entete(fi)
    base_out, h_out = entete(fo)

    import os
    taille = os.path.getsize(chemin_out)
    fin = max(v["data_offsets"][1] for v in h_out.values())
    print(f"tenseurs en sortie : {len(h_out)}")
    print(f"taille  : {taille}")
    print(f"attendu : {base_out + fin}   -> {'OK' if taille == base_out + fin else 'FAUX'}")

    # offsets contigus, sans trou ni recouvrement
    paires = sorted((v["data_offsets"][0], v["data_offsets"][1], k)
                    for k, v in h_out.items())
    curseur, trous = 0, 0
    for a, b, k in paires:
        if a != curseur:
            print(f"  trou/recouvrement avant {k} : {curseur} -> {a}")
            trous += 1
        curseur = b
    print(f"offsets contigus : {'OK' if trous == 0 else str(trous) + ' anomalies'}")

    # ---------------------------------------------------- 3. la reference
    nom = "model.layers.0.self_attn.q_proj.weight"
    att = np.fromfile(ref + "q_proj_row0_attendu.f32", dtype=np.float32)
    o = h_out[nom]["data_offsets"][0]
    got = np.frombuffer(lire(fo, base_out + o, att.nbytes), dtype=np.float32)
    div = int((att.view(np.uint32) != got.view(np.uint32)).sum())
    print(f"ligne 0 de q_proj : {div} divergence(s) sur {att.size}  "
          f"-> {'OK' if div == 0 else 'FAUX'}")

    # ---------------------------------------------------- 4/5. chaque tenseur
    def attendu(nom, lo, hi):
        """Recalcule les octets [lo, hi) du tenseur de sortie, depuis l'entree."""
        e = h_in[nom]
        a, b = e["data_offsets"]
        if e["dtype"] != "U8":                     # branche recopie
            return lire(fi, base_in + a + lo, hi - lo)
        # branche decodage : 4 octets de sortie <- 1 quartet d'entree
        e0, e1 = lo // 4, hi // 4                  # indices d'element
        paq = np.frombuffer(lire(fi, base_in + a + e0 // 2, (e1 - e0) // 2),
                            dtype=np.uint8)
        codes = np.empty(e1 - e0, dtype=np.uint8)
        codes[0::2], codes[1::2] = paq & 15, paq >> 4
        sa, sb = h_in[nom.replace(".weight", ".weight_scale")]["data_offsets"]
        ech = E4M3[np.frombuffer(lire(fi, base_in + sa + e0 // 16, (e1 - e0) // 16),
                                 dtype=np.uint8)]
        ga = h_in[nom.replace(".weight", ".weight_scale_2")]["data_offsets"][0]
        glob = np.frombuffer(lire(fi, base_in + ga, 4), dtype=np.float32)[0]
        tot = (glob * np.repeat(ech, 16)).astype(np.float32)
        return (tot * E2M1[codes]).astype(np.float32).tobytes()

    faux, verifies, octets = [], 0, 0
    for nom, v in h_out.items():
        out_a, out_b = v["data_offsets"]
        taille_t = out_b - out_a
        pas = 64                                   # 16 elements : bloc entier
        fenetre = min(4096, (taille_t // pas) * pas) or taille_t

        points = {0, max(0, taille_t - fenetre)}   # debut et fin
        k = CHUNK                                  # bords des morceaux
        while k < taille_t:
            points.add(max(0, (k - fenetre // 2) // pas * pas))
            k += CHUNK
        for lo in sorted(points):
            hi = min(lo + fenetre, taille_t)
            lo = (lo // pas) * pas
            if hi - lo < pas:
                continue
            got = lire(fo, base_out + out_a + lo, hi - lo)
            exp = attendu(nom, lo, hi)
            octets += hi - lo
            if got != exp:
                d = next(i for i in range(len(got)) if got[i] != exp[i])
                faux.append((nom, lo, d))
                break
        verifies += 1

    print(f"\ntenseurs verifies : {verifies}, {octets / 1e6:.1f} Mo compares")
    if faux:
        print(f"DIVERGENCES : {len(faux)} tenseur(s)")
        for nom, lo, d in faux[:10]:
            print(f"  {nom}  fenetre a {lo}, premier octet faux +{d}")
    else:
        print("0 divergence")


main()
