// JournalDetections : une ligne CSV par défaut détecté, avec la version du modèle qui a décidé.
// Écrit uniquement par le consommateur (un seul thread), d'où l'absence de verrou.

using System.IO;               // non inclus d'office dans un projet WPF
using System.Globalization;

namespace InspectionWpf;

public sealed class JournalDetections : IDisposable
{
    private readonly StreamWriter _w;

    public string Chemin { get; }

    public JournalDetections(string chemin)
    {
        Chemin = chemin;
        _w = new StreamWriter(chemin, append: false) { AutoFlush = true };   // rien de perdu en cas d'arrêt brutal
        _w.WriteLine("horodatage;image;classe;score;x;y;largeur;hauteur;inference_ms;modele");
    }

    public void Ecrire(long idImage, IEnumerable<Detection> detections, string versionModele, double inferenceMs)
    {
        string date = DateTime.Now.ToString("O");
        foreach (var d in detections)
        {
            var b = d.Boite;
            _w.WriteLine(string.Join(';', date, idImage, d.Classe,
                d.Score.ToString("F3", CultureInfo.InvariantCulture),
                b.X, b.Y, b.Width, b.Height,
                inferenceMs.ToString("F1", CultureInfo.InvariantCulture),
                versionModele));
        }
    }

    public void Dispose() => _w.Dispose();
}
