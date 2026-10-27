// Chaîne d'inspection classique en C# (OpenCvSharp), portage des scripts Python etape1 à etape4
//
//   1. Prétraitement   : flou médian + correction d'éclairage (division par le fond)
//   2. Seuillage       : seuil statistique à deux côtés (médiane + MAD)
//   3. Nettoyage       : fermeture 3x3 + filtre par surface
//   4. Caractérisation : mesures en mm, indices de forme, classement par règles
//
// Lancer :  dotnet run -- tole.png        (tole.png générée par generer_tole.py)
// Produit : resultat_csharp.png et rapport_csharp.csv

using System.Diagnostics;
using System.Globalization;
using OpenCvSharp;

const double MmParPixel = 0.25;
const double K = 5;                // un défaut s'écarte d'au moins K fois le bruit
const int SurfaceMin = 12;         // pixels (0,75 mm²)
const int MargeBord = 8;           // pixels ignorés sur les bords

string chemin = args.Length > 0 ? args[0] : "tole.png";
using var img = Cv2.ImRead(chemin, ImreadModes.Grayscale);
if (img.Empty())
{
    Console.WriteLine($"Image introuvable : {chemin}");
    return;
}

var chrono = Stopwatch.StartNew();

// ---------- 1. Prétraitement ----------
using var median = new Mat();
Cv2.MedianBlur(img, median, 5);

using var medianF = new Mat();
median.ConvertTo(medianF, MatType.CV_32F);

using var fond = new Mat();
Cv2.GaussianBlur(medianF, fond, new Size(0, 0), 40);

using var corrige = new Mat();
Cv2.Divide(medianF, fond, corrige, scale: 128);            // image / fond * 128

// ---------- 2. Seuillage statistique ----------
corrige.GetArray(out float[] valeurs);
double niveau = Mediane(valeurs);
double bruit = 1.4826 * Mediane(valeurs.Select(v => (float)Math.Abs(v - niveau)).ToArray());

using var ecart = new Mat();
Cv2.Absdiff(corrige, new Scalar(niveau), ecart);           // |image - niveau|

using var masqueF = new Mat();
Cv2.Threshold(ecart, masqueF, K * bruit, 255, ThresholdTypes.Binary);
using var masque = new Mat();
masqueF.ConvertTo(masque, MatType.CV_8U);

// ---------- 3. Nettoyage : fermeture 3x3, puis filtre par surface pendant l'étiquetage ----------
using var noyau = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
Cv2.MorphologyEx(masque, masque, MorphTypes.Close, noyau);

using var etiquettes = new Mat();
using var stats = new Mat();
using var centres = new Mat();
int n = Cv2.ConnectedComponentsWithStats(masque, etiquettes, stats, centres);

// ---------- 4. Caractérisation ----------
var defauts = new List<Defaut>();
for (int i = 1; i < n; i++)                                // 0 = le fond
{
    int x = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
    int y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
    int w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
    int h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);
    int aire = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);

    if (aire < SurfaceMin) continue;                                         // bruit
    if (x < MargeBord || y < MargeBord || x + w > img.Width - MargeBord || y + h > img.Height - MargeBord)
        continue;                                                            // bord de l'image

    using var zone = new Mat();
    Cv2.Compare(etiquettes, new Scalar(i), zone, CmpTypes.EQ);               // 255 sur ce défaut

    Cv2.FindContours(zone, out Point[][] contours, out _, RetrievalModes.External,
                     ContourApproximationModes.ApproxNone);
    var contour = contours.OrderByDescending(c => Cv2.ContourArea(c)).First();

    RotatedRect rect = Cv2.MinAreaRect(contour);
    double longueur = Math.Max(rect.Size.Width, rect.Size.Height);
    double largeur = Math.Min(rect.Size.Width, rect.Size.Height);
    double perimetre = Cv2.ArcLength(contour, true);

    double surfaceMm2 = aire * MmParPixel * MmParPixel;
    double diametreEq = 2 * Math.Sqrt(surfaceMm2 / Math.PI);
    double allongement = longueur / Math.Max(largeur, 1);
    double circularite = 4 * Math.PI * aire / (perimetre * perimetre);
    bool sombre = Cv2.Mean(corrige, zone).Val0 < niveau;

    string classe = allongement > 5 ? "rayure"
                  : sombre && diametreEq >= 5 ? "tache"
                  : sombre ? "inclusion"
                  : "inconnu";

    defauts.Add(new Defaut(defauts.Count + 1, classe, rect,
        surfaceMm2, longueur * MmParPixel, largeur * MmParPixel, diametreEq, allongement, circularite,
        sombre ? "sombre" : "clair"));
}

chrono.Stop();

// ---------- Résultats ----------
Console.WriteLine($"Niveau du métal {niveau:F0}, bruit {bruit:F1}, seuils {niveau - K * bruit:F0} et {niveau + K * bruit:F0}");
Console.WriteLine($"{defauts.Count} défauts trouvés en {chrono.ElapsedMilliseconds} ms\n");
foreach (var d in defauts)
    Console.WriteLine($"{d.Id,2} {d.Classe,-10} surface {d.SurfaceMm2,7:F2} mm²  longueur {d.LongueurMm,6:F1} mm  " +
                      $"largeur {d.LargeurMm,5:F2} mm  diam. éq. {d.DiametreEqMm,6:F2} mm  allong. {d.Allongement,5:F1}  {d.Polarite}");

// Rapport CSV (traçabilité)
var fr = CultureInfo.InvariantCulture;
var lignes = new List<string> { "id;classe;centre_x_mm;centre_y_mm;surface_mm2;longueur_mm;largeur_mm;diametre_eq_mm;allongement;circularite;polarite" };
lignes.AddRange(defauts.Select(d => string.Join(";",
    d.Id, d.Classe,
    (d.Rect.Center.X * MmParPixel).ToString("F1", fr), (d.Rect.Center.Y * MmParPixel).ToString("F1", fr),
    d.SurfaceMm2.ToString("F2", fr), d.LongueurMm.ToString("F1", fr), d.LargeurMm.ToString("F2", fr),
    d.DiametreEqMm.ToString("F2", fr), d.Allongement.ToString("F1", fr), d.Circularite.ToString("F2", fr), d.Polarite)));
File.WriteAllLines("rapport_csharp.csv", lignes);

// Image annotée
using var vis = new Mat();
Cv2.CvtColor(img, vis, ColorConversionCodes.GRAY2BGR);
var couleurs = new Dictionary<string, Scalar>
{
    ["rayure"] = new Scalar(255, 170, 0),
    ["tache"] = new Scalar(60, 90, 255),
    ["inclusion"] = new Scalar(40, 200, 60),
    ["inconnu"] = new Scalar(200, 200, 200),
};
foreach (var d in defauts)
{
    var coins = d.Rect.Points().Select(p => new Point((int)p.X, (int)p.Y)).ToArray();
    Cv2.Polylines(vis, new[] { coins }, true, couleurs[d.Classe], 2);
    string texte = d.Classe == "rayure" ? $"{d.Id} rayure L={d.LongueurMm:F0} mm" : $"{d.Id} {d.Classe} d={d.DiametreEqMm:F1} mm";
    var haut = new Point(coins.Min(p => p.X), Math.Max(coins.Min(p => p.Y) - 8, 20));
    Cv2.PutText(vis, texte, haut, HersheyFonts.HersheySimplex, 0.8, couleurs[d.Classe], 2);
}
Cv2.ImWrite("resultat_csharp.png", vis);
Console.WriteLine("\nresultat_csharp.png et rapport_csharp.csv créés");


static double Mediane(float[] tableau)
{
    var copie = (float[])tableau.Clone();
    Array.Sort(copie);
    return copie[copie.Length / 2];
}

record Defaut(int Id, string Classe, RotatedRect Rect, double SurfaceMm2, double LongueurMm, double LargeurMm,
              double DiametreEqMm, double Allongement, double Circularite, string Polarite);