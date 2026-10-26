"""
Le seuillage, séparer les défauts du métal sain

On compare quatre façons de faire :
  A. Seuil fixe sur l'image BRUTE           -> piège de l'éclairage inégal
  B. Otsu (seuil automatique) sur l'image corrigée -> piège de l'histogramme à une seule bosse
  C. Seuil adaptatif sur l'image corrigée    -> compare chaque pixel à ses voisins
  D. Seuil statistique à deux côtés sur l'image corrigée :
     un pixel est un défaut s'il s'écarte du métal sain de plus de k fois le bruit,
     vers le clair (rayure) OU vers le sombre (tache, inclusion)

Lancer :  python etape2_seuillage.py   (après generer_tole.py)
Produit : etape2_resultat.png et masque_defauts.png (utilisé à l'étape 3)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tole.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit("tole.png introuvable : exécuter d'abord generer_tole.py")

# ---------- Prétraitement de l'étape 1 ----------
median = cv2.medianBlur(img, 5).astype(np.float32)
fond = cv2.GaussianBlur(median, (0, 0), sigmaX=40)
corrige = np.clip(median / fond * 128, 0, 255).astype(np.uint8)


def pourcentage(masque):
    """Part de l'image marquée comme défaut. Les vrais défauts couvrent environ 0,3 % de l'image."""
    return 100 * np.count_nonzero(masque) / masque.size


# ---------- A. Seuil fixe sur l'image brute ----------
# "Tout ce qui est plus sombre que 70 est un défaut sombre"
_, masque_a = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV)

# ---------- B. Otsu sur l'image corrigée ----------
# Otsu cherche LE seuil qui sépare le mieux deux groupes de pixels.
# Il suppose un histogramme à deux bosses (fond + objet).
seuil_otsu, masque_b = cv2.threshold(corrige, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ---------- C. Seuil adaptatif sur l'image corrigée ----------
# Chaque pixel est comparé à la moyenne de ses voisins (fenêtre 51 x 51) moins une marge C.
# Ici : défauts sombres seulement.
masque_c = cv2.adaptiveThreshold(corrige, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=51, C=15)

# ---------- D. Seuil statistique à deux côtés ----------
# Niveau du métal sain et bruit, mesurés de façon robuste (médiane et MAD :
# ils ne se laissent pas influencer par les quelques pixels de défauts)
niveau = np.median(corrige)
mad = np.median(np.abs(corrige.astype(np.float32) - niveau))
bruit = 1.4826 * mad          # le facteur 1,4826 convertit la MAD en écart-type équivalent
k = 5                          # un défaut s'écarte d'au moins 5 fois le bruit
ecart = np.abs(corrige.astype(np.float32) - niveau)
masque_d = np.where(ecart > k * bruit, 255, 0).astype(np.uint8)

print(f"Métal sain : niveau {niveau:.0f}, bruit {bruit:.1f} -> seuils {niveau - k * bruit:.0f} et {niveau + k * bruit:.0f}")
print(f"Seuil d'Otsu trouvé : {seuil_otsu:.0f}")
print()
print("Part de l'image marquée comme défaut (vrais défauts ≈ 0,3 %) :")
for nom, m in [("A. Seuil fixe, image brute", masque_a),
               ("B. Otsu, image corrigée", masque_b),
               ("C. Adaptatif, image corrigée", masque_c),
               ("D. Statistique 2 côtés, corrigée", masque_d)]:
    print(f"  {nom:35s} {pourcentage(m):6.2f} %")

cv2.imwrite("masque_defauts.png", masque_d)

# ---------- Figure ----------
fig, ax = plt.subplots(3, 2, figsize=(13, 11))

ax[0, 0].imshow(corrige, cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_title("Image corrigée (étape 1)")

ax[0, 1].hist(corrige.ravel(), bins=256, range=(0, 255), color="gray")
ax[0, 1].set_yscale("log")
ax[0, 1].axvline(seuil_otsu, color="red", ls="--", label=f"Otsu = {seuil_otsu:.0f}")
ax[0, 1].axvline(niveau - k * bruit, color="green", ls="--", label="seuils statistiques (D)")
ax[0, 1].axvline(niveau + k * bruit, color="green", ls="--")
ax[0, 1].set_title("Histogramme (échelle log) : une seule grosse bosse")
ax[0, 1].set_xlabel("niveau de gris")
ax[0, 1].legend()

for a, m, titre in [
    (ax[1, 0], masque_a, f"A. Seuil fixe 70 sur l'image brute ({pourcentage(masque_a):.1f} %)"),
    (ax[1, 1], masque_b, f"B. Otsu sur l'image corrigée ({pourcentage(masque_b):.1f} %)"),
    (ax[2, 0], masque_c, f"C. Adaptatif, défauts sombres ({pourcentage(masque_c):.2f} %)"),
    (ax[2, 1], masque_d, f"D. Statistique 2 côtés, k = {k} ({pourcentage(masque_d):.2f} %)"),
]:
    a.imshow(m, cmap="gray")
    a.set_title(titre)

for a in ax.ravel():
    if a is not ax[0, 1]:
        a.axis("off")

plt.tight_layout()
plt.savefig("etape2_resultat.png", dpi=90)
print("\netape2_resultat.png et masque_defauts.png créés")
