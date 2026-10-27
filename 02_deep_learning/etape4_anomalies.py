"""
La détection d'anomalies, quand on n'a (presque) pas d'images de défauts

Principe (version simplifiée de PatchCore) :
  1. On prend un réseau DÉJÀ entraîné sur ImageNet (transfer learning), sans le réentraîner.
  2. On lui montre UNIQUEMENT des tôles saines. Pour chaque petit morceau (patch) de l'image,
     il produit un vecteur de caractéristiques : sa "description" du morceau.
  3. On range toutes ces descriptions dans une "banque de mémoire" : ce qu'est une tôle normale.
  4. Sur une nouvelle image, chaque patch est comparé à son plus proche voisin dans la banque.
     Loin de tout ce qui est normal = anomalie.

Deux expériences :
  A. Banque construite avec des tôles saines "simples"
  B. Banque construite avec des tôles saines qui ont AUSSI des marquages et des ombres
     (des éléments normaux sur la vraie ligne)

Lancer :  python etape4_anomalies.py
Produit : etape4_resultat.png
Le réseau yolo11n-cls.pt (5 Mo) est téléchargé automatiquement la première fois.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO

H, W = 608, 1216                 # taille de la tôle (multiples de 16 pour le réseau)
torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # carte graphique si disponible
print(f"Calcul sur : {DEVICE}")


# ======================================================================
# Les images
# ======================================================================
def tole_saine(rng, marquage=False, ombre=False):
    """Une tôle sans défaut, avec des variations NORMALES : éclairage, stries, bruit."""
    img = np.full((H, W), rng.uniform(130, 150))
    img += rng.uniform(3, 7) * np.sin(2 * np.pi * np.arange(H)[:, None] / 6.0 + rng.uniform(0, 6.28))
    if marquage:                                          # numéro de bobine, à une position au hasard
        m = np.zeros((H, W), np.uint8)
        texte = "".join(rng.choice(list("ABCDEFGH0123456789"), 5))
        cv2.putText(m, texte, (int(rng.uniform(40, 760)), int(rng.uniform(120, 560))),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 6)
        img[m > 0] = rng.uniform(85, 105)
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    a, b = rng.uniform(-0.4, 0.4, 2)
    eclairage = 0.8 + a * (xx - 0.5) + b * (yy - 0.5)
    if ombre:                                             # ombre nette d'un support, à une position au hasard
        x0 = int(rng.uniform(80, W - 120))
        eclairage[:, x0:x0 + 40] *= rng.uniform(0.65, 0.8)
    img = img * eclairage + rng.normal(0, 6, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8)


def tole_difficile():
    """La tôle difficile de etape6_limites.py : 3 vrais défauts et 2 pièges."""
    rng = np.random.default_rng(7)
    img = np.full((600, 1200), 140.0)
    img += 6 * np.sin(2 * np.pi * np.arange(600)[:, None] / 6.0)
    cv2.line(img, (120, 120), (420, 160), 200, 2, cv2.LINE_AA)                 # rayure
    pli = np.zeros_like(img)
    cv2.line(pli, (650, 80), (1000, 230), 1.0, 9, cv2.LINE_AA)
    img -= 9 * cv2.GaussianBlur(pli, (0, 0), 3)                                 # pli léger
    for _ in range(25):                                                         # nid de piqûres
        cv2.circle(img, (int(rng.normal(300, 25)), int(rng.normal(430, 18))), 1, 90, -1)
    m = np.zeros((600, 1200), np.uint8)
    cv2.putText(m, "B4127", (760, 470), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 6)
    img[m > 0] = 95                                                             # marquage (normal)
    xx, _ = np.meshgrid(np.linspace(0, 1, 1200), np.linspace(0, 1, 600))
    ecl = 0.6 + 0.4 * xx
    ecl[:, 560:600] *= 0.72                                                     # ombre (normale)
    img = np.clip(img * ecl + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return cv2.copyMakeBorder(img, 0, H - 600, 0, W - 1200, cv2.BORDER_REPLICATE)


# ======================================================================
# Le réseau pré-entraîné, utilisé comme "descripteur" de patchs
# ======================================================================
reseau = YOLO("yolo11n-cls.pt").model.model.eval().to(DEVICE)


@torch.no_grad()
def caracteristiques(img):
    """Image (H, W) -> carte de vecteurs (192, H/4, W/4) : une description par patch de 4 x 4 px."""
    x = torch.from_numpy(img).float().div(255).expand(1, 3, H, W).contiguous().to(DEVICE)
    sorties = {}
    for i, couche in enumerate(reseau):
        x = couche(x)
        if i in (2, 4):                     # couche 2 : détails fins (1/4) ; couche 4 : contexte (1/8)
            sorties[i] = x
        if i == 4:
            break
    c4 = F.interpolate(sorties[4], size=sorties[2].shape[-2:], mode="bilinear", align_corners=False)
    feat = torch.cat([sorties[2], c4], dim=1)
    return F.avg_pool2d(feat, 3, stride=1, padding=1)[0]  # chaque patch "voit" aussi ses voisins


@torch.no_grad()
def banque(images, par_image=3000, taille=10000):
    """Construit la mémoire. Un tirage au hasard raterait les patchs RARES (un marquage occupe
    très peu de surface) : on utilise donc une sélection "coreset", qui garde des patchs
    les plus différents possible les uns des autres, pour couvrir toute la variété du normal."""
    morceaux = []
    for im in images:
        v = caracteristiques(im).flatten(1).T
        morceaux.append(v[torch.randperm(len(v), device=v.device)[:par_image]])
    candidats = torch.cat(morceaux)

    # Sélection gloutonne : on ajoute à chaque fois le patch le PLUS LOIN de ceux déjà choisis.
    # (calculée sur une projection aléatoire en 64 dimensions, pour aller plus vite)
    proj = candidats @ torch.randn(candidats.shape[1], 64, device=candidats.device) / 8
    choisis = [0]
    dist_min = ((proj - proj[0]) ** 2).sum(1)
    for _ in range(taille - 1):
        suivant = int(dist_min.argmax())
        choisis.append(suivant)
        dist_min = torch.minimum(dist_min, ((proj - proj[suivant]) ** 2).sum(1))
    return candidats[choisis]


@torch.no_grad()
def carte_anomalie(img, memoire):
    f = caracteristiques(img)
    c, h, w = f.shape
    patchs = f.flatten(1).T
    d = torch.cat([torch.cdist(bloc, memoire).min(dim=1).values      # distance au plus proche voisin normal
                   for bloc in patchs.split(2048)])                    # par blocs, pour ne pas saturer la RAM
    carte = d.reshape(1, 1, h, w)
    carte = F.interpolate(carte, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return cv2.GaussianBlur(carte, (0, 0), 4)


# ======================================================================
# Les deux expériences
# ======================================================================
rng = np.random.default_rng(0)
saines_A = [tole_saine(rng) for _ in range(40)]
valid_A = [tole_saine(rng) for _ in range(6)]
saines_B = [tole_saine(rng, marquage=(i % 2 == 0), ombre=(i % 3 == 0)) for i in range(40)]
valid_B = [tole_saine(rng, marquage=(i % 2 == 0), ombre=(i % 3 == 0)) for i in range(6)]
test = tole_difficile()

zones = {  # (x0, y0, x1, y1)
    "rayure (défaut)": (110, 105, 430, 175), "pli léger (défaut)": (640, 70, 1010, 240),
    "nid de piqûres (défaut)": (220, 370, 380, 490), "marquage (normal)": (750, 400, 1040, 490),
    "ombre (normal)": (540, 0, 620, 600),
}

resultats = {}
for nom, saines, validation in [("A", saines_A, valid_A), ("B", saines_B, valid_B)]:
    memoire = banque(saines)
    seuil = np.percentile(np.concatenate([carte_anomalie(v, memoire).ravel() for v in validation]), 99.9)
    carte = carte_anomalie(test, memoire)
    resultats[nom] = (carte, seuil)
    print(f"\nExpérience {nom} : seuil = {seuil:.2f} (99,9 % des pixels de tôles saines de validation sont en dessous)")
    for zone, (x0, y0, x1, y1) in zones.items():
        score = carte[y0:y1, x0:x1].max()
        print(f"  {zone:24s} score max {score:5.2f}  -> {'ANOMALIE' if score > seuil else 'normal'}")

# ======================================================================
# Figure
# ======================================================================
fig, ax = plt.subplots(3, 1, figsize=(11, 10))
ax[0].imshow(test, cmap="gray")
ax[0].set_title("Tôle à inspecter : rayure, pli léger, nid de piqûres + marquage et ombre (normaux)")
for nom, a, titre in [("A", ax[1], "A. Mémoire = tôles saines simples"),
                      ("B", ax[2], "B. Mémoire = tôles saines AVEC marquages et ombres")]:
    carte, seuil = resultats[nom]
    a.imshow(test, cmap="gray")
    a.imshow(np.ma.masked_less(carte, seuil * 0.7), cmap="inferno", alpha=0.75, vmin=seuil * 0.7, vmax=carte.max())
    a.contour(carte > seuil, levels=[0.5], colors="cyan", linewidths=1.5)
    a.set_title(titre + "   (contour cyan = au-dessus du seuil)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("etape4_resultat.png", dpi=85)
print("\netape4_resultat.png créé")
