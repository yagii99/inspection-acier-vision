"""
Le traitement du signal, la FFT pour supprimer la texture de laminage

Notre tôle a des stries horizontales régulières (une tous les 6 pixels) : c'est la "texture normale"
du métal laminé. Elles ne sont pas des défauts, mais elles ajoutent du "bruit" qui gêne la détection.

  1. En 1D : une colonne de pixels est un signal. Sa FFT montre un pic à la fréquence des stries.
  2. En 2D : le spectre de l'image montre deux points brillants = les stries.
  3. On efface ces deux points (filtre coupe-bande, "notch") et on revient à l'image.
  4. On vérifie : les stries ont disparu, les défauts sont intacts, et le bruit mesuré a baissé.

Lancer :  python etape5_fft.py   (après generer_tole.py)
Produit : etape5_resultat.png
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tole.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit("tole.png introuvable : exécuter d'abord generer_tole.py")

# Correction d'éclairage de l'étape 1, SANS flou médian (il effacerait en partie les stries,
# et on veut justement voir ce que la FFT fait toute seule)
f = img.astype(np.float32)
corrige = f / cv2.GaussianBlur(f, (0, 0), 40) * 128
H, W = corrige.shape

# ======================================================================
# 1. En 1D : une colonne de l'image, vue comme un signal
# ======================================================================
colonne = corrige[:, 1000] - corrige[:, 1000].mean()       # une colonne sans défaut, centrée sur 0
spectre_1d = np.abs(np.fft.rfft(colonne))
freqs_1d = np.fft.rfftfreq(H)                              # en cycles par pixel
pic = freqs_1d[np.argmax(spectre_1d[1:]) + 1]
print(f"1D : pic à {pic:.3f} cycle/pixel, soit une période de {1 / pic:.1f} pixels")

# ======================================================================
# 2. En 2D : le spectre de toute l'image
# ======================================================================
F = np.fft.fftshift(np.fft.fft2(corrige - corrige.mean()))  # fftshift : fréquence nulle au centre
spectre = np.log1p(np.abs(F))                               # log : sinon on ne voit que le pic central

# ======================================================================
# 3. Filtre coupe-bande : on éteint les pics des stries
# ======================================================================
cy, cx = H // 2, W // 2
fy = round(H * pic)                                          # position des pics : à fy lignes du centre
masque_notch = np.ones((H, W), np.float32)
for dy in (-fy, fy):
    cv2.circle(masque_notch, (cx, cy + dy), 6, 0, -1)        # un petit disque noir sur chaque pic
masque_notch = cv2.GaussianBlur(masque_notch, (0, 0), 2)     # bords adoucis : évite des "vagues" parasites

F_filtre = F * masque_notch
filtre = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtre))) + corrige.mean()
retire = corrige - filtre                                    # ce que le filtre a enlevé

# ======================================================================
# 4. Vérifications
# ======================================================================
def bruit_robuste(im):
    m = np.median(im)
    return 1.4826 * np.median(np.abs(im - m))

amplitude_avant = spectre_1d.max() * 2 / H
amplitude_apres = np.abs(np.fft.rfft(filtre[:, 1000] - filtre[:, 1000].mean())).max() * 2 / H
print(f"Amplitude des stries : avant {amplitude_avant:.1f}, après {amplitude_apres:.1f} niveaux de gris")
print(f"Bruit mesuré (MAD)   : avant {bruit_robuste(corrige):.2f}, après {bruit_robuste(filtre):.2f}")
print(f"Contraste de la tache : avant {corrige[375:386, 845:856].mean() - np.median(corrige):+.1f}, "
      f"après {filtre[375:386, 845:856].mean() - np.median(filtre):+.1f}")
print(f"Contraste de la rayure : avant {corrige[180, 345:355].mean() - np.median(corrige):+.1f}, "
      f"après {filtre[180, 345:355].mean() - np.median(filtre):+.1f}")

# ======================================================================
# Figure
# ======================================================================
zoom = (slice(120, 260), slice(250, 520))                    # un morceau avec la rayure
fig, ax = plt.subplots(3, 2, figsize=(13, 12))

ax[0, 0].plot(colonne, lw=0.7)
ax[0, 0].set_xlim(0, 120)
ax[0, 0].set_title("1. Une colonne de pixels = un signal (120 premiers pixels)")
ax[0, 0].set_xlabel("ligne (pixel)")
ax[0, 0].set_ylabel("niveau de gris - moyenne")

ax[0, 1].plot(freqs_1d, spectre_1d, lw=0.8)
ax[0, 1].annotate(f"pic : 1 cycle tous les {1 / pic:.0f} px\n= les stries", (pic, spectre_1d.max()),
                  xytext=(pic + 0.06, spectre_1d.max() * 0.8), arrowprops=dict(arrowstyle="->"))
ax[0, 1].set_title("2. Sa FFT : quelles fréquences contient-il ?")
ax[0, 1].set_xlabel("fréquence (cycles par pixel)")
ax[0, 1].set_ylabel("amplitude")

ax[1, 0].imshow(spectre, cmap="magma")
for dy in (-fy, fy):
    ax[1, 0].add_patch(plt.Circle((cx, cy + dy), 18, fill=False, color="cyan", lw=1.5))
ax[1, 0].set_title("3. Spectre 2D de l'image : les 2 points entourés = les stries")
ax[1, 0].axis("off")

ax[1, 1].imshow(masque_notch, cmap="gray")
ax[1, 1].set_title("4. Filtre coupe-bande : on éteint ces 2 points")
ax[1, 1].axis("off")

ax[2, 0].imshow(corrige[zoom], cmap="gray", vmin=100, vmax=170)
ax[2, 0].set_title("5. Avant (zoom) : stries + rayure")
ax[2, 0].axis("off")

ax[2, 1].imshow(filtre[zoom], cmap="gray", vmin=100, vmax=170)
ax[2, 1].set_title("6. Après (zoom) : stries supprimées, rayure intacte")
ax[2, 1].axis("off")

plt.tight_layout()
plt.savefig("etape5_resultat.png", dpi=85)

plt.figure(figsize=(8, 4))
plt.imshow(retire, cmap="gray")
plt.title("Ce que le filtre a retiré : seulement les stries")
plt.axis("off")
plt.tight_layout()
plt.savefig("etape5_retire.png", dpi=85)
print("\netape5_resultat.png et etape5_retire.png créés")
