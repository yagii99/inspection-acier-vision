"""
La même prédiction en Python, pour comparer avec le programme C#

Lancer :  python comparer_python.py chemin/vers/best.onnx chemin/vers/image.jpg
Les détections doivent être (presque) identiques à celles du C# : mêmes classes,
mêmes scores à 0,01 près, mêmes boîtes à 1 ou 2 pixels près.
"""

import sys
import time

from ultralytics import YOLO


def main():
    modele = YOLO(sys.argv[1], task="detect")       # on charge le MÊME fichier ONNX que le C#
    image = sys.argv[2]
    r = modele.predict(image, imgsz=224, conf=0.25, verbose=False)[0]

    debut = time.perf_counter()
    for _ in range(20):
        modele.predict(image, imgsz=224, conf=0.25, verbose=False)
    print(f"Latence moyenne (prétraitement + inférence + décodage) : {(time.perf_counter() - debut) / 20 * 1000:.1f} ms")

    print(f"{len(r.boxes)} détections :")
    for c, s, b in sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda t: -t[1]):
        print(f"  {r.names[int(c)]:16s} {s:.2f}   x={b[0]:.0f}..{b[2]:.0f}  y={b[1]:.0f}..{b[3]:.0f}")


if __name__ == "__main__":
    main()
