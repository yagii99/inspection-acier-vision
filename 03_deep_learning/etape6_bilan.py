"""
Le modèle entraîné sur NEU-DET fonctionne-t-il sur NOTRE tôle ?

On applique le YOLO entraîné par etape3_entrainer.py (entraîné sur les images NEU-DET) à la tôle difficile générée par 02_traitement_classique/python/etape6_limites.py.
Les images NEU font 200 x 200 px : on découpe donc notre tôle en vignettes de 200 x 200
qui se chevauchent, on fait prédire chaque vignette, puis on remet les boîtes sur la tôle entière.

Lancer :  python etape6_bilan.py                          (modèle par défaut)
          python etape6_bilan.py chemin/vers/best.pt
Produit : etape6_resultat.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def tole_difficile():
    """La tôle difficile de etape6_limites.py : 3 vrais défauts (rayure, pli, nid de piqûres) et 2 pièges."""
    rng = np.random.default_rng(7)
    img = np.full((600, 1200), 140.0)
    img += 6 * np.sin(2 * np.pi * np.arange(600)[:, None] / 6.0)
    cv2.line(img, (120, 120), (420, 160), 200, 2, cv2.LINE_AA)
    pli = np.zeros_like(img)
    cv2.line(pli, (650, 80), (1000, 230), 1.0, 9, cv2.LINE_AA)
    img -= 9 * cv2.GaussianBlur(pli, (0, 0), 3)
    for _ in range(25):
        cv2.circle(img, (int(rng.normal(300, 25)), int(rng.normal(430, 18))), 1, 90, -1)
    m = np.zeros((600, 1200), np.uint8)
    cv2.putText(m, "B4127", (760, 470), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 6)
    img[m > 0] = 95
    xx, _ = np.meshgrid(np.linspace(0, 1, 1200), np.linspace(0, 1, 600))
    ecl = 0.6 + 0.4 * xx
    ecl[:, 560:600] *= 0.72
    return np.clip(img * ecl + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def dernier_modele() -> str:
    """Le best.pt de l'essai le plus récent dans runs/detect."""
    candidats = sorted(Path("runs/detect").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidats:
        raise SystemExit("Aucun modèle trouvé dans runs/detect : exécuter d'abord etape3_entrainer.py")
    return str(candidats[-1])


def main():
    chemin = sys.argv[1] if len(sys.argv) > 1 else dernier_modele()
    modele = YOLO(chemin)
    tole = tole_difficile()
    couleur = cv2.cvtColor(tole, cv2.COLOR_GRAY2BGR)

    T, PAS = 200, 100                                    # vignettes de 200 px, qui se chevauchent de moitié
    boites = []
    for y in range(0, tole.shape[0] - T + 1, PAS):
        for x in range(0, tole.shape[1] - T + 1, PAS):
            vignette = couleur[y:y + T, x:x + T]
            r = modele.predict(vignette, imgsz=224, conf=0.25, verbose=False)[0]
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                boites.append((int(c), s, [b[0] + x, b[1] + y, b[2] + x, b[3] + y]))

    # Les vignettes se chevauchent : on fusionne les boîtes en double (suppression des non-maxima)
    if boites:
        garder = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, _, b in boites],
                                  [s for _, s, _ in boites], 0.25, 0.5)
        boites = [boites[i] for i in np.array(garder).flatten()]

    print(f"{len(boites)} détections sur la tôle difficile :")
    for c, s, b in sorted(boites, key=lambda t: t[2][0]):
        print(f"  {modele.names[c]:16s} confiance {s:.2f}  en x={b[0]:.0f}..{b[2]:.0f}, y={b[1]:.0f}..{b[3]:.0f}")
        cv2.rectangle(couleur, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.putText(couleur, f"{modele.names[c]} {s:.2f}", (int(b[0]), max(int(b[1]) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Rappel de la vérité, en vert
    for nom, (x0, y0, x1, y1) in {"rayure": (110, 105, 430, 175), "pli": (640, 70, 1010, 240),
                                  "piqures": (220, 370, 380, 490)}.items():
        cv2.rectangle(couleur, (x0, y0), (x1, y1), (0, 200, 0), 1)
        cv2.putText(couleur, nom, (x0, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.imwrite("etape6_resultat.png", couleur)
    print("\netape6_resultat.png créé (vert = vrais défauts, rouge = prédictions du modèle NEU)")


if __name__ == "__main__":
    main()
