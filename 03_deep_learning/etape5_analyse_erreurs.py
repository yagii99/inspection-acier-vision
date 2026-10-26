"""
Analyser les erreurs d'un modèle existant (avant de chercher à l'améliorer)

  1. Fait prédire le modèle sur le jeu de TEST
  2. Range chaque prédiction et chaque défaut annoté dans une catégorie :
       - bonne détection          : bonne classe, boîte qui recouvre bien l'annotation
       - mauvaise classe          : bon endroit, mauvais type de défaut
       - fausse alarme            : une boîte là où rien n'est annoté
       - défaut raté              : un défaut annoté que le modèle n'a pas trouvé
  3. Affiche le bilan par classe
  4. Fabrique une galerie des pires erreurs : vert = annotation, rouge = prédiction
  5. Cherche un seuil de confiance PAR CLASSE sur la validation, et mesure le gain sur le test

Lancer :  python etape5_analyse_erreurs.py                      (modèle par défaut)
          python etape5_analyse_erreurs.py chemin/vers/best.pt
Produit : etape5_galerie_fausses_alarmes.jpg, etape5_galerie_ratés.jpg
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IOU_MIN = 0.5            # recouvrement minimal pour dire qu'une boîte « tombe » sur l'annotation
CONF_AFFICHAGE = 0.25    # seuil de confiance utilisé pour l'analyse


def iou(a, b):
    """Intersection sur union de deux boîtes (x0, y0, x1, y1)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def lire_annotations(chemin_label, w, h):
    boites = []
    if chemin_label.exists():
        for ligne in chemin_label.read_text().split("\n"):
            if ligne.strip():
                c, cx, cy, bw, bh = map(float, ligne.split())
                boites.append((int(c), ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return boites


def predire(modele, dossier_images, conf):
    """Renvoie {nom_image: (image, annotations, prédictions)} ; prédictions = [(classe, score, boîte)]."""
    resultat = {}
    images = sorted(Path(dossier_images).glob("*.jpg"))
    for r in modele.predict([str(p) for p in images], conf=conf, imgsz=224, verbose=False, stream=True):
        p = Path(r.path)
        h, w = r.orig_shape
        label = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
        preds = [(int(c), float(s), tuple(b)) for c, s, b in
                 zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        resultat[p.name] = (r.orig_img, lire_annotations(label, w, h), preds)
    return resultat


def apparier(annotations, predictions, seuils=None):
    """Associe chaque prédiction à au plus une annotation. Renvoie la liste des événements."""
    if seuils is not None:
        predictions = [p for p in predictions if p[1] >= seuils[p[0]]]
    predictions = sorted(predictions, key=lambda p: -p[1])          # les plus sûres d'abord
    prises = set()
    evenements = []
    for c, s, b in predictions:
        meilleur, j_meilleur = 0, None
        for j, (ca, ba) in enumerate(annotations):
            if j not in prises and iou(b, ba) > meilleur:
                meilleur, j_meilleur = iou(b, ba), j
        if j_meilleur is not None and meilleur >= IOU_MIN:
            prises.add(j_meilleur)
            ca = annotations[j_meilleur][0]
            evenements.append(("bonne détection" if ca == c else "mauvaise classe", ca if ca == c else c, s, b, ca))
        else:
            evenements.append(("fausse alarme", c, s, b, None))
    for j, (ca, ba) in enumerate(annotations):
        if j not in prises:
            evenements.append(("défaut raté", ca, 0.0, ba, ca))
    return evenements


def galerie(cas, noms, titre_fichier, n=12):
    """cas = [(image, annotations, prédictions, texte)] -> une mosaïque de n images annotées."""
    vignettes = []
    for img, annots, preds, texte in cas[:n]:
        v = img.copy()
        for c, b in annots:
            cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
        for c, s, b in preds:
            cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
        v = cv2.resize(v, (300, 300))
        cv2.rectangle(v, (0, 0), (300, 24), (0, 0, 0), -1)
        cv2.putText(v, texte, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vignettes.append(v)
    if not vignettes:
        return
    while len(vignettes) % 4:
        vignettes.append(np.zeros_like(vignettes[0]))
    lignes = [np.hstack(vignettes[i:i + 4]) for i in range(0, len(vignettes), 4)]
    cv2.imwrite(titre_fichier, np.vstack(lignes))


def dernier_modele() -> str:
    """Le best.pt de l'essai le plus récent dans runs/detect."""
    candidats = sorted(Path("runs/detect").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidats:
        raise SystemExit("Aucun modèle trouvé dans runs/detect : exécuter d'abord etape3_entrainer.py")
    return str(candidats[-1])


def main():
    chemin = sys.argv[1] if len(sys.argv) > 1 else dernier_modele()
    modele = YOLO(chemin)
    noms = modele.names

    # ---------- 1-2. Prédictions sur le test et catégories d'erreurs ----------
    test = predire(modele, "neu_yolo/images/test", 0.01)        # on garde tout, on filtre ensuite
    bilan = defaultdict(Counter)
    par_image = {}
    for nom, (img, annots, preds) in test.items():
        evts = apparier(annots, [p for p in preds if p[1] >= CONF_AFFICHAGE])
        par_image[nom] = evts
        for cat, c, *_ in evts:
            bilan[noms[c]][cat] += 1

    # ---------- 3. Bilan par classe ----------
    cats = ["bonne détection", "mauvaise classe", "fausse alarme", "défaut raté"]
    print(f"\nAnalyse sur le TEST (seuil de confiance {CONF_AFFICHAGE}, IoU >= {IOU_MIN})\n")
    print(f"{'classe':18s}" + "".join(f"{c:>17s}" for c in cats))
    for c in noms.values():
        print(f"{c:18s}" + "".join(f"{bilan[c][k]:17d}" for k in cats))
    total = Counter()
    for c in bilan.values():
        total.update(c)
    print(f"{'TOTAL':18s}" + "".join(f"{total[k]:17d}" for k in cats))

    # ---------- 4. Galeries des pires erreurs ----------
    def score_image(nom, cat):
        return sum(1 for e in par_image[nom] if e[0] == cat)

    for cat, fichier in [("fausse alarme", "etape5_galerie_fausses_alarmes.jpg"),
                         ("défaut raté", "etape5_galerie_ratés.jpg")]:
        pires = sorted(test, key=lambda n: -score_image(n, cat))
        cas = []
        for nom in pires:
            k = score_image(nom, cat)
            if k == 0:
                break
            img, annots, preds = test[nom]
            cas.append((img, annots, [p for p in preds if p[1] >= CONF_AFFICHAGE], f"{nom}  {k} x {cat}"))
        galerie(cas, noms, fichier)
        print(f"{fichier} créé ({min(len(cas), 12)} images)")

    # ---------- 5. Un seuil de confiance par classe, choisi sur la VALIDATION ----------
    val = predire(modele, "neu_yolo/images/val", 0.01)
    candidats = np.arange(0.05, 0.95, 0.05)

    def f1(donnees, seuils):
        tp = fp = fn = 0
        for img, annots, preds in donnees.values():
            for cat, *_ in apparier(annots, preds, seuils):
                tp += cat == "bonne détection"
                fp += cat in ("fausse alarme", "mauvaise classe")
                fn += cat == "défaut raté"
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-9), p, r

    seuils = {c: CONF_AFFICHAGE for c in noms}
    for c in noms:                                      # on optimise une classe à la fois
        meilleur = max(candidats, key=lambda s: f1(val, {**seuils, c: s})[0])
        seuils[c] = float(meilleur)

    f_glob = f1(test, {c: CONF_AFFICHAGE for c in noms})
    f_cls = f1(test, seuils)
    print("\nSeuils par classe choisis sur la validation :")
    for c, s in seuils.items():
        print(f"  {noms[c]:18s} {s:.2f}")
    print(f"\nSur le TEST : seuil unique {CONF_AFFICHAGE} -> F1 {f_glob[0]:.3f} (P {f_glob[1]:.3f}, R {f_glob[2]:.3f})")
    print(f"              seuils par classe -> F1 {f_cls[0]:.3f} (P {f_cls[1]:.3f}, R {f_cls[2]:.3f})")

    # Fichier lu par l'application C# (Detecteur) s'il est posé à côté du modèle ONNX
    sortie = Path("seuils_par_classe.json")
    sortie.write_text(json.dumps({noms[c]: round(s, 2) for c, s in seuils.items()}, indent=2), encoding="utf-8")
    print(f"\nSeuils enregistrés dans {sortie}")


if __name__ == "__main__":
    main()
