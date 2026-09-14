"""
Oracle NVFP4 — regenere les fichiers de reference et retranche la convention
de quartets, a partir des deux checkpoints publics.

    python oracle.py

Ne telecharge que quelques Ko (requetes HTTP d'intervalle), pas les 8 Go.
Dependances : torch (pour float8_e4m3fn), numpy.
"""

import json
import struct
import urllib.request

import numpy as np
import torch

# ---------------------------------------------------------------- checkpoints

NVFP4_REPO = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
NVFP4_FILE = "model-00001-of-00002.safetensors"

BF16_REPO = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # miroir non restreint
BF16_FILE = "model-00001-of-00004.safetensors"

TENSEUR = "model.layers.0.self_attn.q_proj.weight"

# a lancer depuis la racine du depot : python outils/oracle.py
REF = "reference/"

# les 16 valeurs E2M1, indexees par le code
E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


def url(repo, fichier):
    return f"https://huggingface.co/{repo}/resolve/main/{fichier}"


def intervalle(u, debut, fin):
    """Requete HTTP d'intervalle : octets [debut, fin)."""
    req = urllib.request.Request(u, headers={"Range": f"bytes={debut}-{fin - 1}"})
    return urllib.request.urlopen(req).read()


def entete(u):
    """Rend (offset du blob de donnees, en-tete JSON)."""
    n = struct.unpack("<Q", intervalle(u, 0, 8))[0]
    return 8 + n, json.loads(intervalle(u, 8, 8 + n))


def main():
    # ------------------------------------------------------ cote NVFP4
    u4 = url(NVFP4_REPO, NVFP4_FILE)
    base4, h4 = entete(u4)
    print(f"en-tete NVFP4 : {base4 - 8} octets, {len(h4)} entrees")

    off_w = h4[TENSEUR]["data_offsets"][0]
    off_s = h4[TENSEUR.replace(".weight", ".weight_scale")]["data_offsets"][0]
    off_g = h4[TENSEUR.replace(".weight", ".weight_scale_2")]["data_offsets"][0]

    paquet = intervalle(u4, base4 + off_w, base4 + off_w + 2048)  # ligne 0
    ech_br = intervalle(u4, base4 + off_s, base4 + off_s + 256)   # 256 blocs
    glob_br = intervalle(u4, base4 + off_g, base4 + off_g + 4)

    globale = np.float32(struct.unpack("<f", glob_br)[0])
    echelles = torch.frombuffer(bytearray(ech_br), dtype=torch.float8_e4m3fn)
    echelles = echelles.float().numpy().astype(np.float32)
    print(f"echelle globale = {globale!r}")

    # ------------------------------------------------------ cote BF16 d'origine
    ub = url(BF16_REPO, BF16_FILE)
    baseb, hb = entete(ub)
    off_b = hb[TENSEUR]["data_offsets"][0]
    bf16 = torch.frombuffer(
        bytearray(intervalle(ub, baseb + off_b, baseb + off_b + 8192)),
        dtype=torch.bfloat16,
    ).float().numpy().astype(np.float32)

    print(f"shape NVFP4 declaree : {h4[TENSEUR]['shape']}  (dtype "
          f"{h4[TENSEUR]['dtype']})")
    print(f"shape BF16 declaree  : {hb[TENSEUR]['shape']}  <- la vraie")

    # --------------------------------- la question : quel quartet est l'indice pair ?
    octets = np.frombuffer(paquet, dtype=np.uint8)
    bas, haut = octets & 15, octets >> 4

    resultats = {}
    for nom, (pair, impair) in [
        ("bas = indice pair", (bas, haut)),
        ("haut = indice pair", (haut, bas)),
    ]:
        codes = np.empty(4096, dtype=np.uint8)
        codes[0::2], codes[1::2] = pair, impair
        # Ordre d'association volontaire : on replie les deux echelles en une
        # seule, une fois par bloc, puis une multiplication par valeur. C'est
        # l'ordre du decodeur Zig -- deux fois moins de multiplications, au prix
        # d'un arrondi supplementaire (1 ULP sur ~6 % des valeurs, soit ~4e-8
        # en relatif, contre 10 % d'erreur de quantification).
        echelle_totale = (globale * np.repeat(echelles, 16)).astype(np.float32)
        val = (echelle_totale * E2M1[codes]).astype(np.float32)
        r = float(np.corrcoef(val, bf16)[0, 1])
        resultats[nom] = (r, val)
        print(f"  {nom:22} correlation avec l'original = {r: .6f}")

    gagnant = max(resultats, key=lambda k: resultats[k][0])
    ref = resultats[gagnant][1].astype(np.float32)
    print(f"\n=> convention retenue : {gagnant}")

    ecart = float(np.abs(ref - bf16).mean() / np.abs(bf16).mean())
    print(f"=> ecart relatif moyen NVFP4 vs BF16 : {ecart:.2%}")

    # ------------------------------------------------------ fichiers d'oracle
    open(REF + "q_proj_row0_packed.bin", "wb").write(paquet)
    open(REF + "q_proj_row0_scales.bin", "wb").write(ech_br)
    open(REF + "q_proj_global_scale.bin", "wb").write(glob_br)
    ref.tofile(REF + "q_proj_row0_attendu.f32")
    bf16.tofile(REF + "q_proj_row0_bf16_origine.f32")
    print("\nfichiers d'oracle reecrits.")


if __name__ == "__main__":
    main()
