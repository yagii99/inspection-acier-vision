// JournalDetections : une ligne CSV par défaut détecté, avec la version du modèle qui a décidé,
// et une ligne NON_INSPECTEE pour chaque image qui n'a pas pu être traitée.
// Deux threads écrivent (le consommateur pour les défauts, le producteur pour les zones non inspectées) :
// un verrou garantit que les lignes ne se mélangent pas.

using System.IO;               // non inclus d'office dans un projet WPF
using System.Globalization;

namespace InspectionWpf;

public sealed class JournalDetections : IDisposable
{
    private readonly StreamWriter _w;
    private readonly object _verrou = new();

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
        lock (_verrou)
        {
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
    }

    /// <summary>Image reçue mais pas inspectée : la zone correspondante doit être recontrôlée.</summary>
    public void EcrireNonInspectee(long idImage, DateTime horodatageUtc, string versionModele)
    {
        lock (_verrou)
        {
            _w.WriteLine(string.Join(';', horodatageUtc.ToLocalTime().ToString("O"), idImage, "NON_INSPECTEE",
                "", "", "", "", "", "", versionModele));
        }
    }

    public void Dispose()
    {
        lock (_verrou) { _w.Dispose(); }
    }
}
