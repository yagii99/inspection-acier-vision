"""
Préparer le dataset NEU-DET pour YOLO

NEU-DET : 1 800 images de 200 x 200 pixels de défauts de surface d'acier laminé à chaud,
6 classes (300 images chacune), annotées avec des boîtes au format Pascal VOC (un .xml par image).

Ce script :
  1. lit les annotations .xml
  2. les convertit au format YOLO : un .txt par image, une ligne par boîte
       classe  x_centre  y_centre  largeur  hauteur     (toutes les valeurs entre 0 et 1)
  3. découpe en train / val / test (70 / 15 / 15 %), classe par classe, pour garder l'équilibre
  4. écrit le fichier neu.yaml que YOLO utilise pour savoir où sont les données

Lancer :  python etape3_preparer_neu.py  CHEMIN_VERS_NEU
          (le dossier qui contient IMAGES/ et ANNOTATIONS/, et éventuellement
           Validation_Images/ et Validation_Annotations/)
Produit : le dossier neu_yolo/ et le fichier neu.yaml
"""

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
# En français : craquelures, inclusions, plaques, surface piquée, calamine incrustée, rayures

source = Path(sys.argv[1] if len(sys.argv) > 1 else "NEU-DET")
sortie = Path("neu_yolo")
random.seed(0)                                   # même découpage à chaque lancement : reproductible

# ---------- 1. Retrouver toutes les paires image + annotation ----------
paires = []
for dossier_img, dossier_xml in [("IMAGES", "ANNOTATIONS"), ("Validation_Images", "Validation_Annotations")]:
    for xml in sorted((source / dossier_xml).glob("*.xml")):
        img = source / dossier_img / (xml.stem + ".jpg")
        if img.exists():
            paires.append((img, xml))
if not paires:
    raise SystemExit(f"Aucune image trouvée dans {source.resolve()} (attendu : IMAGES/ et ANNOTATIONS/)")


# ---------- 2. Conversion VOC -> YOLO ----------
def vers_yolo(xml_path):
    racine = ET.parse(xml_path).getroot()
    W = float(racine.find("size/width").text)
    H = float(racine.find("size/height").text)
    lignes = []
    for obj in racine.iter("object"):
        nom = obj.find("name").text.strip()
        boite = obj.find("bndbox")
        x0, y0, x1, y1 = (float(boite.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)   # on reste dans l'image
        if x1 <= x0 or y1 <= y0:
            continue                                                     # boîte invalide : ignorée
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        bw, bh = (x1 - x0) / W, (y1 - y0) / H
        lignes.append(f"{CLASSES.index(nom)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lignes


# ---------- 3. Découpage train / val / test, classe par classe ----------
par_classe = {c: [] for c in CLASSES}
for img, xml in paires:
    classe = next(c for c in CLASSES if img.stem.startswith(c))           # la classe est dans le nom du fichier
    par_classe[classe].append((img, xml))

if sortie.exists():
    shutil.rmtree(sortie)
compteur = Counter()
boites = Counter()
for classe, liste in par_classe.items():
    random.shuffle(liste)
    n = len(liste)
    coupes = {"train": liste[: int(0.7 * n)], "val": liste[int(0.7 * n): int(0.85 * n)], "test": liste[int(0.85 * n):]}
    for partie, elements in coupes.items():
        (sortie / "images" / partie).mkdir(parents=True, exist_ok=True)
        (sortie / "labels" / partie).mkdir(parents=True, exist_ok=True)
        for img, xml in elements:
            lignes = vers_yolo(xml)
            shutil.copy(img, sortie / "images" / partie / img.name)
            (sortie / "labels" / partie / (img.stem + ".txt")).write_text("\n".join(lignes) + "\n")
            compteur[(partie, classe)] += 1
            boites[partie] += len(lignes)

# ---------- 4. Le fichier de configuration pour YOLO ----------
Path("neu.yaml").write_text(
    f"path: {sortie.resolve().as_posix()}\n"
    "train: images/train\nval: images/val\ntest: images/test\n"
    "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
    encoding="utf-8")

print(f"{len(paires)} images converties\n")
print(f"{'classe':18s}{'train':>7s}{'val':>6s}{'test':>6s}")
for c in CLASSES:
    print(f"{c:18s}{compteur[('train', c)]:7d}{compteur[('val', c)]:6d}{compteur[('test', c)]:6d}")
print(f"\nBoîtes : train {boites['train']}, val {boites['val']}, test {boites['test']}")
print("neu_yolo/ et neu.yaml créés")
