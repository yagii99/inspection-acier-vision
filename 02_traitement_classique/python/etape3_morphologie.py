"""
La morphologie, nettoyer le masque
  Partie 1 : les 4 opérations sur une image jouet, pour voir ce que fait chacune
  Partie 2 : nettoyage du vrai masque de l'étape 2 (masque_defauts.png)
             + le piège : un noyau trop gros efface la rayure fine
             + l'alternative : filtrer les morceaux par leur surface

Lancer :  python etape3_morphologie.py   (après etape2_seuillage.py)
Produit : etape3_operations.png, etape3_resultat.png, masque_propre.png (utilisé à l'étape 4)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Partie 1 : image jouet
# ======================================================================
jouet = np.zeros((120, 200), np.uint8)
cv2.rectangle(jouet, (30, 30), (110, 90), 255, -1)       # un objet
cv2.circle(jouet, (55, 60), 1, 0, -1)                   # un petit trou (3 px)
cv2.circle(jouet, (90, 60), 6, 0, -1)                   # un gros trou (13 px)
cv2.line(jouet, (125, 20), (190, 100), 255, 2)          # une ligne fine (2 px)
rng = np.random.default_rng(1)
for _ in range(40):                                     # des points de bruit isolés
    x, y = rng.integers(0, 200), rng.integers(0, 120)
    jouet[y, x] = 255

noyau = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
ops = [
    ("Original", jouet),
    ("Érosion : rogne les bords", cv2.erode(jouet, noyau)),
    ("Dilatation : épaissit les bords", cv2.dilate(jouet, noyau)),
    ("Ouverture = érosion puis dilatation\n(supprime les petits points)", cv2.morphologyEx(jouet, cv2.MORPH_OPEN, noyau)),
    ("Fermeture = dilatation puis érosion\n(bouche les petits trous)", cv2.morphologyEx(jouet, cv2.MORPH_CLOSE, noyau)),
    ("Ouverture 3x3 puis fermeture 15x15\n(noyau plus gros : bouche le gros trou)", cv2.morphologyEx(cv2.morphologyEx(jouet, cv2.MORPH_OPEN, noyau), cv2.MORPH_CLOSE,
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))),
]
fig, ax = plt.subplots(2, 3, figsize=(13, 6))
for a, (titre, im) in zip(ax.ravel(), ops):
    a.imshow(im, cmap="gray")
    a.set_title(titre, fontsize=10)
    a.axis("off")
plt.tight_layout()
plt.savefig("etape3_operations.png", dpi=85)

# ======================================================================
# Partie 2 : le vrai masque
# ======================================================================
masque = cv2.imread("masque_defauts.png", cv2.IMREAD_GRAYSCALE)
if masque is None:
    raise SystemExit("masque_defauts.png introuvable : exécuter d'abord etape2_seuillage.py")


def morceaux(m):
    """Nombre de morceaux blancs séparés (composantes connexes), sans compter le fond."""
    n, _ = cv2.connectedComponents(m)
    return n - 1


def surface_rayure(m):
    """Pixels blancs dans la zone de la rayure : pour voir si elle survit."""
    return int(np.count_nonzero(m[140:220, 140:560]))


petit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
gros = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

ouvert_petit = cv2.morphologyEx(masque, cv2.MORPH_OPEN, petit)
ouvert_ferme = cv2.morphologyEx(ouvert_petit, cv2.MORPH_CLOSE, petit)
ouvert_gros = cv2.morphologyEx(masque, cv2.MORPH_OPEN, gros)

SURFACE_MIN = 12                       # en pixels : 12 px à 0,25 mm/px = 0,75 mm²


def filtre_surface(m, surface_min):
    """Garde les morceaux d'au moins surface_min pixels, sans déformer leur forme."""
    n, etiquettes, stats, _ = cv2.connectedComponentsWithStats(m)
    sortie = np.zeros_like(m)
    for i in range(1, n):              # 0 = le fond
        if stats[i, cv2.CC_STAT_AREA] >= surface_min:
            sortie[etiquettes == i] = 255
    return sortie


def trous(m):
    """Nombre de trous noirs entièrement entourés de blanc (dans les défauts)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(m))
    h, w = m.shape
    return sum(1 for i in range(1, n)
               if stats[i, 0] > 0 and stats[i, 1] > 0
               and stats[i, 0] + stats[i, 2] < w and stats[i, 1] + stats[i, 3] < h)


par_surface = filtre_surface(masque, SURFACE_MIN)

# La combinaison retenue : une petite fermeture pour boucher les trous dans les défauts,
# puis le filtre par surface pour supprimer le bruit sans toucher aux rayures fines
propre = filtre_surface(cv2.morphologyEx(masque, cv2.MORPH_CLOSE, petit), SURFACE_MIN)

print(f"{'':40s}{'morceaux':>9s}{'trous':>7s}{'pixels rayure':>15s}")
for nom, m in [("Masque de l'étape 2", masque),
               ("Ouverture 3x3 + fermeture 3x3", ouvert_ferme),
               ("Ouverture 7x7 (noyau trop gros)", ouvert_gros),
               (f"Filtre par surface (>= {SURFACE_MIN} px)", par_surface),
               ("Fermeture 3x3 + filtre par surface", propre)]:
    print(f"{nom:40s}{morceaux(m):9d}{trous(m):7d}{surface_rayure(m):15d}")

cv2.imwrite("masque_propre.png", propre)

fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
for a, m, titre in [
    (ax[0, 0], masque, f"Masque de l'étape 2 : {morceaux(masque)} morceaux"),
    (ax[0, 1], ouvert_ferme, f"Ouverture 3x3 + fermeture 3x3 : {morceaux(ouvert_ferme)} morceaux"),
    (ax[1, 0], ouvert_gros, f"Ouverture 7x7 : la rayure a disparu ({morceaux(ouvert_gros)} morceaux)"),
    (ax[1, 1], propre, f"Fermeture 3x3 + filtre par surface : {morceaux(propre)} morceaux"),
]:
    a.imshow(m, cmap="gray")
    a.set_title(titre)
    a.axis("off")
plt.tight_layout()
plt.savefig("etape3_resultat.png", dpi=85)
print("\netape3_operations.png, etape3_resultat.png et masque_propre.png créés")
