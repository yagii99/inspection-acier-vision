// Detecteur : inférence YOLO (ONNX Runtime) sur des images Mono8.
// Même prétraitement que Ultralytics (letterbox, RGB, /255, NCHW), résultats validés contre Python.
//
// Formats de sortie pris en charge :
//   [1, 4 + nc, N]  YOLOv8 / YOLO11 / YOLO26 exportés sans NMS : décodage + NMS ici
//   [1, N, 6]       export end-to-end (x1, y1, x2, y2, score, classe) : déjà filtré par le modèle
//
//   using var detecteur = new Detecteur("best.onnx");          // une fois, au démarrage
//   var defauts = detecteur.Detecter(pixels, largeur, hauteur); // à chaque image

using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace InspectionWpf;

public sealed record Detection(string Classe, float Score, Rect Boite);

public sealed class Detecteur : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _nomEntree;
    private readonly int _taille;

    public string[] Classes { get; }
    public string Version { get; }                 // journalisée avec chaque décision (traçabilité)
    public float SeuilConfiance { get; set; } = 0.25f;
    public float SeuilNms { get; set; } = 0.7f;

    /// <summary>Seuils par classe (choisis sur la validation). Une classe absente utilise SeuilConfiance.</summary>
    public Dictionary<string, float> SeuilsParClasse { get; } = new();

    public Detecteur(string cheminOnnx)
    {
        _session = new InferenceSession(cheminOnnx);
        var entree = _session.InputMetadata.First();
        _nomEntree = entree.Key;
        _taille = entree.Value.Dimensions[2] > 0 ? entree.Value.Dimensions[2] : 224;   // imgsz utilisé à l'export
        Classes = LireNomsDesClasses(_session);
        var fichier = new FileInfo(cheminOnnx);
        Version = $"{fichier.Name} ({fichier.LastWriteTime:yyyy-MM-dd HH:mm}, {_taille} px)";
    }

    public List<Detection> Detecter(byte[] pixels, int largeur, int hauteur)
    {
        using var gris = new Mat(hauteur, largeur, MatType.CV_8UC1);
        Marshal.Copy(pixels, 0, gris.Data, largeur * hauteur);
        using var bgr = new Mat();
        Cv2.CvtColor(gris, bgr, ColorConversionCodes.GRAY2BGR);       // le modèle attend 3 canaux

        var (tenseur, echelle, dx, dy) = Letterbox(bgr, _taille);
        using var resultats = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_nomEntree, tenseur) });
        var sortie = resultats.First().AsTensor<float>();

        var brutes = sortie.Dimensions[2] == 6
            ? DecoderEndToEnd(sortie, echelle, dx, dy)
            : DecoderAvecNms(sortie, echelle, dx, dy);

        var cadre = new Rect(0, 0, largeur, hauteur);
        return brutes
            .Where(d => d.Score >= SeuilsParClasse.GetValueOrDefault(d.Classe, SeuilConfiance))
            .Select(d => d with { Boite = d.Boite.Intersect(cadre) })
            .OrderByDescending(d => d.Score)
            .ToList();
    }

    private List<Detection> DecoderAvecNms(Tensor<float> s, float echelle, float dx, float dy)
    {
        int nc = s.Dimensions[1] - 4, n = s.Dimensions[2];
        float seuilMin = SeuilsParClasse.Count > 0 ? Math.Min(SeuilConfiance, SeuilsParClasse.Values.Min()) : SeuilConfiance;
        var boites = new List<Rect>();
        var scores = new List<float>();
        var ids = new List<int>();

        for (int i = 0; i < n; i++)
        {
            int c = 0;
            float score = 0;
            for (int k = 0; k < nc; k++)
                if (s[0, 4 + k, i] > score) { score = s[0, 4 + k, i]; c = k; }
            if (score < seuilMin) continue;

            float cx = (s[0, 0, i] - dx) / echelle, cy = (s[0, 1, i] - dy) / echelle;
            float w = s[0, 2, i] / echelle, h = s[0, 3, i] / echelle;
            boites.Add(new Rect((int)Math.Round(cx - w / 2), (int)Math.Round(cy - h / 2), (int)Math.Round(w), (int)Math.Round(h)));
            scores.Add(score);
            ids.Add(c);
        }

        // NMS par classe : un décalage par classe empêche deux classes différentes de se supprimer
        var decalees = boites.Select((b, k) => new Rect(b.X + ids[k] * 4096, b.Y + ids[k] * 4096, b.Width, b.Height)).ToList();
        CvDnn.NMSBoxes(decalees, scores, seuilMin, SeuilNms, out int[] gardees);
        return gardees.Select(k => new Detection(Classes[ids[k]], scores[k], boites[k])).ToList();
    }

    private List<Detection> DecoderEndToEnd(Tensor<float> s, float echelle, float dx, float dy)
    {
        var liste = new List<Detection>();
        for (int i = 0; i < s.Dimensions[1]; i++)
        {
            float score = s[0, i, 4];
            if (score <= 0) continue;
            int x1 = (int)Math.Round((s[0, i, 0] - dx) / echelle), y1 = (int)Math.Round((s[0, i, 1] - dy) / echelle);
            int x2 = (int)Math.Round((s[0, i, 2] - dx) / echelle), y2 = (int)Math.Round((s[0, i, 3] - dy) / echelle);
            liste.Add(new Detection(Classes[(int)s[0, i, 5]], score, new Rect(x1, y1, x2 - x1, y2 - y1)));
        }
        return liste;
    }

    // Letterbox : redimensionner sans déformer, compléter en gris 114, puis BGR -> RGB, /255, HWC -> CHW
    private static (DenseTensor<float>, float, float, float) Letterbox(Mat bgr, int taille)
    {
        float echelle = Math.Min((float)taille / bgr.Width, (float)taille / bgr.Height);
        int nw = (int)Math.Round(bgr.Width * echelle), nh = (int)Math.Round(bgr.Height * echelle);
        float dx = (taille - nw) / 2f, dy = (taille - nh) / 2f;

        using var redim = new Mat();
        Cv2.Resize(bgr, redim, new Size(nw, nh), 0, 0, InterpolationFlags.Linear);
        using var carre = new Mat();
        Cv2.CopyMakeBorder(redim, carre, (int)Math.Round(dy - 0.1), (int)Math.Round(dy + 0.1),
                           (int)Math.Round(dx - 0.1), (int)Math.Round(dx + 0.1),
                           BorderTypes.Constant, new Scalar(114, 114, 114));
        using var rgb = new Mat();
        Cv2.CvtColor(carre, rgb, ColorConversionCodes.BGR2RGB);
        using var flottant = new Mat();
        rgb.ConvertTo(flottant, MatType.CV_32FC3, 1.0 / 255);

        var donnees = new float[3 * taille * taille];
        Mat[] plans = Cv2.Split(flottant);
        for (int c = 0; c < 3; c++)
        {
            Marshal.Copy(plans[c].Data, donnees, c * taille * taille, taille * taille);
            plans[c].Dispose();
        }
        return (new DenseTensor<float>(donnees, new[] { 1, 3, taille, taille }), echelle, dx, dy);
    }

    private static string[] LireNomsDesClasses(InferenceSession session)
    {
        if (session.ModelMetadata.CustomMetadataMap.TryGetValue("names", out string? texte))
        {
            var noms = Regex.Matches(texte, @"(\d+)\s*:\s*'([^']*)'")
                            .OrderBy(m => int.Parse(m.Groups[1].Value))
                            .Select(m => m.Groups[2].Value)
                            .ToArray();
            if (noms.Length > 0) return noms;
        }
        int n = session.OutputMetadata.First().Value.Dimensions[1] - 4;
        return Enumerable.Range(0, n).Select(i => $"classe {i}").ToArray();
    }

    public void Dispose() => _session.Dispose();
}
