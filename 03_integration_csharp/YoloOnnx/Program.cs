// Inférence d'un modèle YOLO exporté en ONNX, en C#, sans dépendance Python
//
//   1. Charger le modèle ONNX avec ONNX Runtime (aucun Python nécessaire en production)
//   2. Prétraiter l'image comme Ultralytics : letterbox à la taille du modèle, RGB, valeurs 0..1, NCHW
//   3. Lancer l'inférence
//   4. Décoder la sortie : [1, 4 + nb_classes, N propositions] -> boîtes, classes, scores
//   5. Supprimer les boîtes en double (NMS), remettre les boîtes à l'échelle de l'image d'origine
//   6. Mesurer la latence moyenne
//
// Lancer :  dotnet run -- chemin\vers\best.onnx chemin\vers\image.jpg
// Produit : resultat_onnx.jpg

using System.Diagnostics;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

const float SeuilConfiance = 0.25f;
const float SeuilNms = 0.7f;      // valeur par défaut d'Ultralytics

if (args.Length < 2)
{
    Console.WriteLine("Usage : dotnet run -- best.onnx image.jpg");
    return;
}

using var session = new InferenceSession(args[0]);   // créée UNE seule fois au démarrage de l'application
string nomEntree = session.InputMetadata.Keys.First(); // "images" pour un modèle Ultralytics
int taille = session.InputMetadata[nomEntree].Dimensions[2];   // imgsz utilisé à l'export (224, 320...)
var classes = LireNomsDesClasses(session);
Console.WriteLine($"Modèle chargé : entrée « {nomEntree} » {taille} px, {classes.Length} classes : {string.Join(", ", classes)}");

using var image = Cv2.ImRead(args[1], ImreadModes.Color);
if (image.Empty())
{
    Console.WriteLine($"Image introuvable : {args[1]}");
    return;
}

// ---------- 2. Prétraitement ----------
var (entree, echelle, decalX, decalY) = Pretraiter(image, taille);

// ---------- 3. Inférence (+ mesure de la latence) ----------
var valeurs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(nomEntree, entree) };
using (session.Run(valeurs)) { }                      // 1er passage "à vide" : initialisation, pas représentatif

const int Repetitions = 20;
var chrono = Stopwatch.StartNew();
for (int i = 0; i < Repetitions - 1; i++)
    using (session.Run(valeurs)) { }
using var resultats = session.Run(valeurs);
chrono.Stop();
var sortie = resultats.First().AsTensor<float>();     // forme [1, 4 + nb_classes, N]

// ---------- 4. Décodage ----------
int nbClasses = sortie.Dimensions[1] - 4;
int nbPropositions = sortie.Dimensions[2];
var boites = new List<Rect>();
var scores = new List<float>();
var ids = new List<int>();

for (int i = 0; i < nbPropositions; i++)
{
    // La meilleure classe pour cette proposition (les scores sont déjà entre 0 et 1)
    int meilleure = 0;
    float score = 0;
    for (int c = 0; c < nbClasses; c++)
    {
        float s = sortie[0, 4 + c, i];
        if (s > score) { score = s; meilleure = c; }
    }
    if (score < SeuilConfiance) continue;

    // Boîte (centre, largeur, hauteur) en pixels de l'image 224 x 224 -> pixels de l'image d'origine
    float cx = (sortie[0, 0, i] - decalX) / echelle;
    float cy = (sortie[0, 1, i] - decalY) / echelle;
    float w = sortie[0, 2, i] / echelle;
    float h = sortie[0, 3, i] / echelle;
    boites.Add(new Rect((int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h));
    scores.Add(score);
    ids.Add(meilleure);
}

// ---------- 5. NMS par classe : on décale les boîtes de chaque classe pour qu'elles ne se suppriment pas entre elles ----------
var decalees = boites.Select((b, k) => new Rect(b.X + ids[k] * 4096, b.Y + ids[k] * 4096, b.Width, b.Height)).ToList();
CvDnn.NMSBoxes(decalees, scores, SeuilConfiance, SeuilNms, out int[] gardees);

Console.WriteLine($"\nLatence moyenne de l'inférence : {chrono.Elapsed.TotalMilliseconds / Repetitions:F1} ms ({Repetitions} passages)");
Console.WriteLine($"{gardees.Length} détections :");
foreach (int k in gardees.OrderByDescending(k => scores[k]))
{
    var b = boites[k].Intersect(new Rect(0, 0, image.Width, image.Height));   // on reste dans l'image, comme Ultralytics
    Console.WriteLine($"  {classes[ids[k]],-16} {scores[k]:F2}   x={b.X}..{b.X + b.Width}  y={b.Y}..{b.Y + b.Height}");
    Cv2.Rectangle(image, b, new Scalar(0, 0, 255), 2);
    Cv2.PutText(image, $"{classes[ids[k]]} {scores[k]:F2}", new Point(b.X, Math.Max(b.Y - 4, 12)),
                HersheyFonts.HersheySimplex, 0.4, new Scalar(0, 0, 255), 1);
}
Cv2.ImWrite("resultat_onnx.jpg", image);
Console.WriteLine("\nresultat_onnx.jpg créé");


// ======================================================================
// Prétraitement identique à Ultralytics : letterbox (redimensionner sans déformer, puis compléter en gris 114)
// ======================================================================
static (DenseTensor<float>, float, float, float) Pretraiter(Mat image, int cote)
{
    float echelle = Math.Min((float)cote / image.Width, (float)cote / image.Height);
    int nw = (int)Math.Round(image.Width * echelle), nh = (int)Math.Round(image.Height * echelle);
    float decalX = (cote - nw) / 2f, decalY = (cote - nh) / 2f;

    using var redim = new Mat();
    Cv2.Resize(image, redim, new Size(nw, nh), 0, 0, InterpolationFlags.Linear);
    using var carre = new Mat();
    Cv2.CopyMakeBorder(redim, carre, (int)Math.Round(decalY - 0.1), (int)Math.Round(decalY + 0.1),
                       (int)Math.Round(decalX - 0.1), (int)Math.Round(decalX + 0.1),
                       BorderTypes.Constant, new Scalar(114, 114, 114));

    var tenseur = new DenseTensor<float>(new[] { 1, 3, cote, cote });
    var pixels = carre.GetGenericIndexer<Vec3b>();
    for (int y = 0; y < cote; y++)
        for (int x = 0; x < cote; x++)
        {
            Vec3b p = pixels[y, x];               // OpenCV range les couleurs en BGR
            tenseur[0, 0, y, x] = p.Item2 / 255f; // R
            tenseur[0, 1, y, x] = p.Item1 / 255f; // G
            tenseur[0, 2, y, x] = p.Item0 / 255f; // B
        }
    return (tenseur, echelle, decalX, decalY);
}

// Ultralytics enregistre les noms des classes dans les métadonnées du fichier ONNX : {0: 'crazing', 1: 'inclusion', ...}
static string[] LireNomsDesClasses(InferenceSession session)
{
    if (session.ModelMetadata.CustomMetadataMap.TryGetValue("names", out string? texte))
    {
        var noms = Regex.Matches(texte, @"(\d+)\s*:\s*'([^']*)'")
                        .Cast<Match>()
                        .OrderBy(m => int.Parse(m.Groups[1].Value))
                        .Select(m => m.Groups[2].Value)
                        .ToArray();
        if (noms.Length > 0) return noms;
    }
    int n = session.OutputMetadata.First().Value.Dimensions[1] - 4;
    return Enumerable.Range(0, n).Select(i => $"classe {i}").ToArray();
}
