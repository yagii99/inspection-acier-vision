"""
Les limites du traitement classique

On reprend EXACTEMENT la chaîne des étapes 1 à 4 (sans rien régler de nouveau) et on l'applique
à une tôle plus réaliste, avec quatre pièges qu'on rencontre vraiment en usine :
  1. une ombre nette (un support devant l'éclairage)       -> ce n'est pas un défaut
  2. un marquage imprimé normal (numéro de bobine)          -> ce n'est pas un défaut
  3. un pli léger, peu contrasté                             -> c'est un défaut
  4. un nid de piqûres : beaucoup de petits points proches   -> c'est un défaut

Lancer :  python etape6_limites.py
Produit : etape6_resultat.png
"""

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

MM = 0.25
H, W = 600, 1200
rng = np.random.default_rng(7)

# ======================================================================
# La tôle "difficile"
# ======================================================================
img = np.full((H, W), 140.0)
img += 6 * np.sin(2 * np.pi * np.arange(H)[:, None] / 6.0)          # texture de laminage

# Vrais défauts
cv2.line(img, (120, 120), (420, 160), 200, 2, cv2.LINE_AA)            # rayure (comme avant)
pli = np.zeros_like(img)
cv2.line(pli, (650, 80), (1000, 230), 1.0, 9, cv2.LINE_AA)
img -= 9 * cv2.GaussianBlur(pli, (0, 0), 3)                           # pli léger : -9 niveaux, flou
for _ in range(25):                                                   # nid de piqûres
    x, y = int(rng.normal(300, 25)), int(rng.normal(430, 18))
    cv2.circle(img, (x, y), 1, 90, -1)

# Pièges (pas des défauts)
marq = np.zeros((H, W), np.uint8)                                    # marquage normal (putText veut du 8 bits)
cv2.putText(marq, "B4127", (760, 470), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 6)
img[marq > 0] = 95
xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
eclairage = 0.6 + 0.4 * xx
eclairage[:, 560:600] *= 0.72                                         # ombre nette d'un support
img *= eclairage
img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

verite = [  # (nom, est_un_defaut, zone x0, y0, x1, y1)
    ("rayure", True, 110, 105, 430, 175),
    ("pli léger", True, 640, 70, 1010, 240),
    ("nid de piqûres", True, 220, 370, 380, 490),
    ("marquage B4127", False, 750, 400, 1040, 490),
    ("ombre du support", False, 520, 0, 640, 600),
]

# ======================================================================
# La chaîne classique des étapes 1 à 4, à l'identique
# ======================================================================
median = cv2.medianBlur(img, 5).astype(np.float32)
corrige = np.clip(median / cv2.GaussianBlur(median, (0, 0), 40) * 128, 0, 255)
niveau = np.median(corrige)
bruit = 1.4826 * np.median(np.abs(corrige - niveau))
masque = (np.abs(corrige - niveau) > 5 * bruit).astype(np.uint8) * 255
masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
n, etiq, stats, centres = cv2.connectedComponentsWithStats(masque)

detections = []
for i in range(1, n):
    x, y, w, h, aire = stats[i]
    if aire < 12 or x < 8 or y < 8 or x + w > W - 8 or y + h > H - 8:
        continue
    cnt = max(cv2.findContours((etiq == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
              key=cv2.contourArea)
    (cx, cy), (l1, l2), _ = cv2.minAreaRect(cnt)
    allong = max(l1, l2) / max(min(l1, l2), 1)
    sombre = corrige[etiq == i].mean() < niveau
    deq = 2 * math.sqrt(aire * MM * MM / math.pi)
    classe = "rayure" if allong > 5 else ("tache" if sombre and deq >= 5 else ("inclusion" if sombre else "inconnu"))
    detections.append({"classe": classe, "boite": (x, y, w, h), "centre": (cx, cy)})

print(f"Bruit mesuré : {bruit:.1f}  -> seuil = ± {5 * bruit:.0f} niveaux autour de {niveau:.0f}")
print(f"{len(detections)} détections\n")


def dans(c, z):
    return z[0] <= c[0] <= z[2] and z[1] <= c[1] <= z[3]


bilan = []
for nom, est_defaut, *z in verite:
    trouves = [d for d in detections if dans(d["centre"], z)]
    if est_defaut:
        etat = "détecté" if trouves else "RATÉ (faux négatif)"
        if nom == "nid de piqûres" and trouves:
            etat = f"vu en {len(trouves)} morceaux séparés, sans comprendre que c'est un seul défaut"
    else:
        etat = f"FAUSSE ALARME ({len(trouves)} détections)" if trouves else "ignoré, correct"
    bilan.append((nom, est_defaut, etat, z))
    print(f"  {nom:18s} {'défaut' if est_defaut else 'normal':7s} -> {etat}")

# ======================================================================
# Figure
# ======================================================================
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for d in detections:
    x, y, w, h = d["boite"]
    cv2.rectangle(vis, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 200, 0), 2)
for nom, est_defaut, etat, z in bilan:
    bon = ("détecté" in etat) or ("correct" in etat)
    couleur = (60, 200, 60) if bon else (40, 40, 255)
    cv2.rectangle(vis, (z[0], z[1]), (z[2], z[3]), couleur, 2, cv2.LINE_4)
    cv2.putText(vis, nom, (z[0] + 4, max(z[1] - 8, 20) if z[1] > 20 else z[3] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)

fig, ax = plt.subplots(1, 2, figsize=(17, 5.2))
ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
ax[0].set_title("Cadres fins cyan = détections de la chaîne classique   |   vert = bon, rouge = erreur")
ax[0].axis("off")
ax[1].imshow(masque, cmap="gray")
ax[1].set_title("Masque après seuillage et nettoyage (étapes 2 et 3)")
ax[1].axis("off")
plt.tight_layout()
plt.savefig("etape6_resultat.png", dpi=80)
print("\netape6_resultat.png créé")
