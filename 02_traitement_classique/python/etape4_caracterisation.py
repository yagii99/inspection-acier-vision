"""
Contours et caractérisation des défauts
  1. Retrouver chaque défaut dans le masque propre (composantes connexes + contours)
  2. Le mesurer en millimètres : surface, longueur, largeur, angle, diamètre équivalent
  3. Calculer des indices de forme : allongement, circularité, polarité (clair ou sombre)
  4. Le classer avec des règles simples
  5. Comparer à la vérité terrain, et enregistrer un rapport CSV (traçabilité)

Lancer :  python etape4_caracterisation.py   (après etape3_morphologie.py)
Produit : etape4_resultat.png et rapport_defauts.csv
"""

import csv
import json
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tole.png", cv2.IMREAD_GRAYSCALE)
masque = cv2.imread("masque_propre.png", cv2.IMREAD_GRAYSCALE)
if img is None or masque is None:
    raise SystemExit("Exécuter d'abord generer_tole.py, etape2_seuillage.py et etape3_morphologie.py")

with open("verite_terrain.json", encoding="utf-8") as f:
    verite = json.load(f)
MM = verite["mm_par_pixel"]                    # 0,25 mm par pixel
MARGE_BORD = 8                                 # pixels : bande ignorée sur les bords (correction d'éclairage peu fiable)

# Image corrigée (étape 1) : sert à savoir si un défaut est plus clair ou plus sombre que le métal
median = cv2.medianBlur(img, 5).astype(np.float32)
corrige = np.clip(median / cv2.GaussianBlur(median, (0, 0), 40) * 128, 0, 255)
niveau_sain = float(np.median(corrige))

H, W = masque.shape
n, etiquettes, stats, centres = cv2.connectedComponentsWithStats(masque)

defauts = []
for i in range(1, n):
    x, y, w, h, aire_px = stats[i]

    # --- On ignore ce qui touche les bords de l'image ---
    if x < MARGE_BORD or y < MARGE_BORD or x + w > W - MARGE_BORD or y + h > H - MARGE_BORD:
        continue

    zone = (etiquettes == i).astype(np.uint8)
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # --- Mesures géométriques ---
    (cx, cy), (l1, l2), angle = cv2.minAreaRect(contour)   # le plus petit rectangle tourné qui l'entoure
    longueur_px, largeur_px = max(l1, l2), min(l1, l2)
    # Angle du grand côté par rapport à l'horizontale, ramené entre -90 et 90 degrés
    angle_long = angle if l1 >= l2 else angle + 90
    angle = (angle_long + 90) % 180 - 90
    perimetre_px = cv2.arcLength(contour, closed=True)

    surface_mm2 = aire_px * MM * MM
    longueur_mm = longueur_px * MM
    largeur_mm = largeur_px * MM
    diametre_eq_mm = 2 * math.sqrt(surface_mm2 / math.pi)  # diamètre du disque de même surface

    # --- Indices de forme (sans unité) ---
    allongement = longueur_px / max(largeur_px, 1)
    circularite = 4 * math.pi * aire_px / perimetre_px ** 2  # 1 = disque parfait, proche de 0 = ligne

    # --- Polarité : le défaut est-il plus clair ou plus sombre que le métal sain ? ---
    contraste = float(corrige[zone == 1].mean()) - niveau_sain
    polarite = "clair" if contraste > 0 else "sombre"

    # --- Classification par règles ---
    if allongement > 5:
        classe = "rayure"
    elif polarite == "sombre" and diametre_eq_mm >= 5:
        classe = "tache"
    elif polarite == "sombre":
        classe = "inclusion"
    else:
        classe = "inconnu"

    defauts.append({
        "id": len(defauts) + 1, "classe": classe,
        "centre_x_mm": round(cx * MM, 1), "centre_y_mm": round(cy * MM, 1),
        "surface_mm2": round(surface_mm2, 2), "longueur_mm": round(longueur_mm, 1),
        "largeur_mm": round(largeur_mm, 2), "diametre_eq_mm": round(diametre_eq_mm, 2),
        "angle_deg": round(angle, 1), "allongement": round(allongement, 1),
        "circularite": round(circularite, 2), "polarite": polarite, "contraste": round(contraste, 1),
        "_contour": contour, "_rect": cv2.minAreaRect(contour), "_centre_px": (cx, cy),
    })

# ---------- Affichage ----------
print(f"{len(defauts)} défauts trouvés\n")
entete = f"{'id':>2} {'classe':10s} {'surface':>9s} {'longueur':>9s} {'largeur':>8s} {'diam. éq.':>9s} {'allong.':>7s} {'circ.':>6s} {'polarité':>8s}"
print(entete)
for d in defauts:
    print(f"{d['id']:>2} {d['classe']:10s} {d['surface_mm2']:7.2f}mm² {d['longueur_mm']:7.1f}mm {d['largeur_mm']:6.2f}mm "
          f"{d['diametre_eq_mm']:7.2f}mm {d['allongement']:7.1f} {d['circularite']:6.2f} {d['polarite']:>8s}")

# ---------- Comparaison à la vérité terrain ----------
print("\nComparaison avec la vérité terrain :")
for v in verite["defauts"]:
    vx, vy = v["centre_px"]
    proche = min(defauts, key=lambda d: math.dist(d["_centre_px"], (vx, vy)), default=None)
    if proche is None or math.dist(proche["_centre_px"], (vx, vy)) > 30:
        print(f"  {v['type']:10s} -> NON DÉTECTÉ")
        continue
    if v["type"] == "rayure":
        reel, mesure, nom = v["longueur_mm"], proche["longueur_mm"], "longueur"
    else:
        reel, mesure, nom = v["diametre_mm"], proche["diametre_eq_mm"], "diamètre"
    ok = "classe OK" if proche["classe"] == v["type"] else f"classe FAUSSE ({proche['classe']})"
    print(f"  {v['type']:10s} -> {nom} réel {reel:6.1f} mm, mesuré {mesure:6.1f} mm "
          f"(écart {100 * (mesure - reel) / reel:+5.1f} %), {ok}")

# ---------- Rapport CSV : une ligne par défaut, pour la traçabilité ----------
champs = [k for k in defauts[0] if not k.startswith("_")] if defauts else []
with open("rapport_defauts.csv", "w", newline="", encoding="utf-8") as f:
    ecrivain = csv.DictWriter(f, fieldnames=champs, delimiter=";")
    ecrivain.writeheader()
    for d in defauts:
        ecrivain.writerow({k: d[k] for k in champs})

# ---------- Image annotée ----------
couleurs = {"rayure": (255, 170, 0), "tache": (60, 90, 255), "inclusion": (40, 200, 60), "inconnu": (200, 200, 200)}
annotee = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for d in defauts:
    c = couleurs[d["classe"]]
    boite = cv2.boxPoints(d["_rect"]).astype(int)
    cv2.drawContours(annotee, [boite], 0, c, 2)
    if d["classe"] == "rayure":
        texte = f"{d['id']} rayure L={d['longueur_mm']:.0f} mm"
    else:
        texte = f"{d['id']} {d['classe']} d={d['diametre_eq_mm']:.1f} mm"
    bx, by = boite[:, 0].min(), boite[:, 1].min()
    cv2.putText(annotee, texte, (int(bx), max(int(by) - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2)

fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [2, 1]})
ax[0].imshow(cv2.cvtColor(annotee, cv2.COLOR_BGR2RGB))
ax[0].set_title("Défauts détectés, mesurés et classés")
ax[0].axis("off")
noms = [f"{d['id']} {d['classe']}" for d in defauts]
ax[1].scatter([d["allongement"] for d in defauts], [d["circularite"] for d in defauts],
              c=[np.array(couleurs[d["classe"]][::-1]) / 255 for d in defauts], s=120)
for d, nom in zip(defauts, noms):
    ax[1].annotate(nom, (d["allongement"], d["circularite"]), textcoords="offset points", xytext=(8, 4))
ax[1].axvline(5, color="gray", ls="--")
ax[1].text(5.3, 0.95, "seuil rayure", color="gray")
ax[1].set_xscale("log")
ax[1].set_xlabel("allongement (longueur / largeur)")
ax[1].set_ylabel("circularité (1 = disque)")
ax[1].set_ylim(0, 1.1)
ax[1].set_title("Les classes se séparent sur la forme")
plt.tight_layout()
plt.savefig("etape4_resultat.png", dpi=100)
print("\netape4_resultat.png et rapport_defauts.csv créés")
