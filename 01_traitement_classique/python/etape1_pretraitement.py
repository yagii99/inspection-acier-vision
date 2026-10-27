"""
Le prétraitement
  A. Débruiter : flou gaussien et flou médian
  B. Corriger l'éclairage : estimer le "fond" avec un très gros flou, puis le retirer

Lancer :  python etape1_pretraitement.py   (après generer_tole.py)
Produit : etape1_resultat.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tole.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit("tole.png introuvable : exécuter d'abord generer_tole.py")

# ---------- A. Débruiter ----------
gaussien = cv2.GaussianBlur(img, (5, 5), 0)   # moyenne pondérée des voisins : lisse tout, bords compris
median = cv2.medianBlur(img, 5)               # valeur médiane des voisins : enlève les points isolés, garde mieux les bords

# ---------- B. Corriger l'éclairage ----------
# Le "fond" = l'éclairage + le métal sans les petits détails.
# Un flou énorme (sigma 40 px) efface défauts et stries, il ne reste que la variation lente de lumière.
fond = cv2.GaussianBlur(median.astype(np.float32), (0, 0), sigmaX=40)

# On DIVISE par le fond (l'éclairage agit comme un multiplicateur), puis on remet à un niveau moyen
corrige = median.astype(np.float32) / fond * 128
corrige = np.clip(corrige, 0, 255).astype(np.uint8)

# ---------- Mesures pour comparer ----------
def niveaux(im):
    gauche = im[:, :200].mean()
    droite = im[:, -200:].mean()
    return gauche, droite

g0, d0 = niveaux(img)
g1, d1 = niveaux(corrige)
print(f"Avant correction : moyenne gauche {g0:.0f}, droite {d0:.0f}  (écart {d0 - g0:.0f})")
print(f"Après correction : moyenne gauche {g1:.0f}, droite {d1:.0f}  (écart {d1 - g1:.0f})")

# Bruit mesuré dans une zone sans défaut (en bas à droite)
zone = (slice(520, 590), slice(1050, 1180))
print(f"Bruit (écart-type) : original {img[zone].std():.1f} | gaussien {gaussien[zone].std():.1f} | médian {median[zone].std():.1f}")

# ---------- Figure ----------
fig, ax = plt.subplots(3, 2, figsize=(13, 11))
for a, im, titre in [
    (ax[0, 0], img, "1. Image brute : éclairage inégal, bruit, stries"),
    (ax[0, 1], median, "2. Après flou médian 5x5"),
    (ax[1, 0], fond, "3. Fond estimé (flou sigma 40) = l'éclairage"),
    (ax[1, 1], corrige, "4. Image corrigée = médian / fond"),
]:
    a.imshow(im, cmap="gray", vmin=0, vmax=255 if im.dtype == np.uint8 else None)
    a.set_title(titre)
    a.axis("off")

# Profils horizontaux : la même ligne avant / après correction
ligne = 300
ax[2, 0].plot(img[ligne], lw=0.6, label="brute")
ax[2, 0].plot(corrige[ligne], lw=0.6, label="corrigée")
ax[2, 0].set_title(f"5. Profil de la ligne {ligne} (gauche → droite)")
ax[2, 0].set_xlabel("colonne (pixel)")
ax[2, 0].set_ylabel("niveau de gris")
ax[2, 0].legend()

ax[2, 1].hist(img.ravel(), bins=128, range=(0, 255), alpha=0.6, label="brute")
ax[2, 1].hist(corrige.ravel(), bins=128, range=(0, 255), alpha=0.6, label="corrigée")
ax[2, 1].set_title("6. Histogrammes : la corrigée est resserrée")
ax[2, 1].set_xlabel("niveau de gris")
ax[2, 1].legend()

plt.tight_layout()
plt.savefig("etape1_resultat.png", dpi=90)
print("etape1_resultat.png créé")
