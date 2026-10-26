"""
Génère une fausse image de tôle, avec les pièges d'une vraie image industrielle :
  - un éclairage inégal (plus sombre à gauche et vers le bas)
  - une texture de laminage (fines stries régulières dans le sens du défilement)
  - du bruit de capteur
  - trois défauts dont on connaît la position et la taille exactes (la "vérité terrain")

Échelle : 0,25 mm par pixel. L'image fait 1200 x 600 pixels, soit 300 x 150 mm de tôle.

Lancer :  python generer_tole.py
Produit : tole.png (l'image) et verite_terrain.json (les vrais défauts)
"""

import json
import cv2
import numpy as np

MM_PAR_PIXEL = 0.25
LARGEUR, HAUTEUR = 1200, 600
rng = np.random.default_rng(42)          # graine fixe : la même image à chaque lancement

# 1. Le métal : un gris moyen
img = np.full((HAUTEUR, LARGEUR), 140.0)

# 2. La texture de laminage : des stries horizontales fines et régulières (période 6 px)
y = np.arange(HAUTEUR)[:, None]
img += 6 * np.sin(2 * np.pi * y / 6.0)

# 3. Les défauts (dessinés AVANT l'éclairage, comme sur une vraie tôle)
defauts = []

# Rayure : une ligne fine et claire, longue, légèrement inclinée
p1, p2 = (150, 150), (550, 210)
cv2.line(img, p1, p2, color=200, thickness=2, lineType=cv2.LINE_AA)
longueur_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
defauts.append({"type": "rayure", "centre_px": [350, 180],
                "longueur_mm": round(longueur_px * MM_PAR_PIXEL, 1)})

# Tache : une zone sombre, ronde et floue (ex. calamine, oxydation)
tache = np.zeros_like(img)
cv2.circle(tache, (850, 380), 28, 1.0, -1)
tache = cv2.GaussianBlur(tache, (0, 0), 6)
img -= 55 * tache
defauts.append({"type": "tache", "centre_px": [850, 380], "diametre_mm": round(56 * MM_PAR_PIXEL, 1)})

# Petite inclusion : un point sombre de 6 px (1,5 mm), proche de la limite de détection
cv2.circle(img, (300, 470), 3, 85, -1)
defauts.append({"type": "inclusion", "centre_px": [300, 470], "diametre_mm": round(6 * MM_PAR_PIXEL, 1)})

# 4. L'éclairage inégal : une "pente" de luminosité (de 55 % à gauche à 100 % à droite)
xx, yy = np.meshgrid(np.linspace(0, 1, LARGEUR), np.linspace(0, 1, HAUTEUR))
eclairage = 0.55 + 0.45 * xx - 0.12 * yy
img *= eclairage

# 5. Le bruit du capteur
img += rng.normal(0, 7, img.shape)

img = np.clip(img, 0, 255).astype(np.uint8)
cv2.imwrite("tole.png", img)

with open("verite_terrain.json", "w", encoding="utf-8") as f:
    json.dump({"mm_par_pixel": MM_PAR_PIXEL, "defauts": defauts}, f, ensure_ascii=False, indent=2)

print("tole.png et verite_terrain.json créés")
for d in defauts:
    print(" ", d)
