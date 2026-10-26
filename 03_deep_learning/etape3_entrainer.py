"""
Entraînement d'un détecteur YOLO sur NEU-DET, évaluation sur le jeu de test et export ONNX.

Chaque essai a son propre dossier (runs/detect/<nom>) et un fichier metriques.json,
pour comparer les modèles entre eux sans écraser les résultats précédents.

Exemples :
    python etape3_entrainer.py                                  # YOLO26n, 224 px
    python etape3_entrainer.py --modele yolo11n.pt              # comparaison avec YOLO11
    python etape3_entrainer.py --modele yolo26s.pt --imgsz 320 --epoques 150

Prérequis : etape3_preparer_neu.py (crée neu_yolo/ et neu.yaml). GPU NVIDIA conseillé.
Sous Windows, le code doit rester dans main() : les workers du DataLoader relancent le script.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modele", default="yolo26n.pt", help="poids pré-entraînés (yolo26n.pt, yolo11n.pt...)")
    p.add_argument("--data", default="neu.yaml")
    p.add_argument("--imgsz", type=int, default=224, help="images NEU en 200 px : inutile d'agrandir beaucoup")
    p.add_argument("--epoques", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--nom", default=None, help="nom de l'essai (par défaut : modèle + taille)")
    return p.parse_args()


def main():
    args = arguments()
    nom = args.nom or f"{Path(args.modele).stem}_{args.imgsz}"

    modele = YOLO(args.modele)
    modele.train(
        data=args.data,
        epochs=args.epoques,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=20,
        seed=0,
        deterministic=True,
        name=nom,
        exist_ok=False,          # un nouvel essai ne remplace jamais le précédent
        # Augmentations limitées à ce qui peut arriver sur une ligne de laminage
        fliplr=0.5,
        flipud=0.5,
        degrees=0.0,             # la bande défile toujours dans le même sens
        hsv_h=0.0, hsv_s=0.0,    # images en niveaux de gris
        hsv_v=0.3,               # variations d'éclairage
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )

    dossier = Path(modele.trainer.save_dir)
    meilleur = YOLO(dossier / "weights" / "best.pt")
    test = meilleur.val(data=args.data, split="test", imgsz=args.imgsz, plots=True,
                        project=str(dossier), name="test")

    metriques = {
        "modele": args.modele,
        "imgsz": args.imgsz,
        "date": datetime.now().isoformat(timespec="seconds"),
        "test": {
            "mAP50": round(float(test.box.map50), 4),
            "mAP50_95": round(float(test.box.map), 4),
            "precision": round(float(test.box.mp), 4),
            "recall": round(float(test.box.mr), 4),
            "AP50_par_classe": {nom_c: round(float(test.box.ap50[i]), 4)
                                for i, nom_c in meilleur.names.items()},
        },
    }
    (dossier / "metriques.json").write_text(json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTest  mAP@0.5 {metriques['test']['mAP50']:.3f}   mAP@0.5:0.95 {metriques['test']['mAP50_95']:.3f}")
    for nom_c, ap in metriques["test"]["AP50_par_classe"].items():
        print(f"  {nom_c:18s}{ap:.3f}")

    onnx = meilleur.export(format="onnx", imgsz=args.imgsz)
    print(f"\nRésultats : {dossier}\nModèle ONNX : {onnx}")


if __name__ == "__main__":
    main()
