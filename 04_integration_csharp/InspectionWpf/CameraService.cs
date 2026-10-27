// CameraService : tout ce qui touche à la caméra ET au modèle, SANS aucune interface graphique.
//
// Le consommateur fait l'inférence (Detecteur, ONNX). L'émulateur peut rejouer un dossier
// d'images (ex. le jeu de test NEU) à cadence fixe. Sans modèle, le traitement est simulé.
//
// Threads :
//   - thread pylon       : SurImageRecue (le PRODUCTEUR)
//   - tâche traitement   : BoucleTraitement (le CONSOMMATEUR : inférence du modèle)
//   - tâche surveillance : reconnexion après une déconnexion
// Les événements (ImageTraitee, Message, EtatChange) sont levés depuis ces threads, jamais depuis le thread UI.

using System.IO;               // non inclus d'office dans un projet WPF
using System.Diagnostics;
using System.Threading.Channels;
using Basler.Pylon;

namespace InspectionWpf;

public enum EtatCamera { Arretee, Connexion, Acquisition, Degrade }

public sealed record Frame(long Id, int Largeur, int Hauteur, byte[] Pixels, DateTime Horodatage);

public sealed record ResultatImage(long Id, int Largeur, int Hauteur, byte[] Pixels, double Moyenne,
                                   TimeSpan Latence, List<Detection> Detections, double InferenceMs);

public readonly record struct Stats(long Recues, long Traitees, long Perdues, long Ratees, long Erreurs, long AvecDefaut);

public sealed class CameraService : IDisposable
{
    private Camera? _camera;
    private Channel<Frame>? _file;
    private CancellationTokenSource? _cts;
    private Task? _traitement;
    private Task? _surveillance;
    private Task? _declenchement;
    private SemaphoreSlim _connexionPerdue = new(0);
    private volatile int _periodeMs = 200;          // déclenchement logiciel : 5 images/s par défaut
    private Detecteur? _detecteur;
    private JournalDetections? _journal;

    private readonly object _verrou = new();
    private bool _connectee;
    private double _exposition = 5000;

    private long _recues, _traitees, _perdues, _ratees, _erreurs, _avecDefaut;
    private volatile int _dureeTraitementMs = 0;

    public event Action<ResultatImage>? ImageTraitee;
    public event Action<string>? Message;
    public event Action<EtatCamera>? EtatChange;

    /// <summary>Chemin du modèle ONNX. S'il est vide ou introuvable, le traitement est simulé.</summary>
    public string CheminModele { get; set; } = "best.onnx";

    /// <summary>Dossier d'images que la caméra émulée va "filmer" (ex. neu_yolo\images\test). Vide = image de test pylon.</summary>
    public string DossierImages { get; set; } = "";

    /// <summary>Durée de traitement simulée, utilisée seulement s'il n'y a pas de modèle.</summary>
    public int DureeTraitementMs
    {
        get => _dureeTraitementMs;
        set => _dureeTraitementMs = value;
    }

    public Stats Statistiques() => new(
        Interlocked.Read(ref _recues), Interlocked.Read(ref _traitees), Interlocked.Read(ref _perdues),
        Interlocked.Read(ref _ratees), Interlocked.Read(ref _erreurs), Interlocked.Read(ref _avecDefaut));

    // ------------------------------------------------------------------
    // Démarrage (appelé hors du thread UI)
    // ------------------------------------------------------------------
    public void Demarrer()
    {
        if (_camera is not null)
            throw new InvalidOperationException("La caméra est déjà démarrée.");

        // Le modèle est chargé une seule fois (c'est lent), avant de démarrer la caméra
        if (_detecteur is null && File.Exists(CheminModele))
        {
            _detecteur = new Detecteur(CheminModele);
            Message?.Invoke($"Modèle chargé : {_detecteur.Version}, classes : {string.Join(", ", _detecteur.Classes)}");
            ChargerSeuils(Path.Combine(Path.GetDirectoryName(Path.GetFullPath(CheminModele))!, "seuils_par_classe.json"));
        }
        else if (_detecteur is null)
            Message?.Invoke($"Modèle introuvable ({CheminModele}) : traitement simulé");

        var infos = CameraFinder.Enumerate();
        if (infos.Count == 0)
            throw new InvalidOperationException("Aucune caméra trouvée (vérifier PYLON_CAMEMU pour l'émulateur).");

        _recues = _traitees = _perdues = _ratees = _erreurs = _avecDefaut = 0;
        _cts = new CancellationTokenSource();
        _connexionPerdue = new SemaphoreSlim(0);

        _file = Channel.CreateBounded<Frame>(
            new BoundedChannelOptions(5) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = true },
            _ => Interlocked.Increment(ref _perdues));

        _camera = new Camera(infos[0]);
        _camera.CameraOpened += Configuration.AcquireContinuous;
        _camera.StreamGrabber.ImageGrabbed += SurImageRecue;
        _camera.ConnectionLost += SurConnexionPerdue;

        try
        {
            lock (_verrou) { OuvrirEtDemarrer(); }
        }
        catch
        {
            _camera.Dispose();
            _camera = null;
            throw;
        }

        _journal = new JournalDetections(Path.Combine(AppContext.BaseDirectory, $"detections_{DateTime.Now:yyyyMMdd_HHmmss}.csv"));
        _traitement = Task.Run(() => BoucleTraitement(_file.Reader));
        _surveillance = Task.Run(() => BoucleSurveillance(_cts.Token));
        _declenchement = Task.Run(() => BoucleDeclenchement(_cts.Token));

        Message?.Invoke($"Caméra ouverte : {_camera.CameraInfo[CameraInfoKey.FriendlyName]}");
        EtatChange?.Invoke(EtatCamera.Acquisition);
    }

    private void OuvrirEtDemarrer()
    {
        _camera!.Open();
        _camera.Parameters[PLCamera.PixelFormat].TrySetValue(PLCamera.PixelFormat.Mono8);

        if (string.IsNullOrWhiteSpace(DossierImages))
            Message?.Invoke("Aucun dossier d'images (INSPECTION_IMAGES vide) : image de test de l'émulateur");
        else if (!Directory.Exists(DossierImages))
            Message?.Invoke($"Dossier d'images introuvable : {DossierImages}");
        else
        {
            // Fonctions propres à l'émulateur : il "filme" les images d'un dossier, l'une après l'autre.
            // Les images NEU font 200 x 200 : on règle la caméra à cette taille.
            _camera.Parameters[PLCamera.Width].TrySetValue(200);
            _camera.Parameters[PLCamera.Height].TrySetValue(200);
            // Ordre : couper l'image de test (prioritaire sur les fichiers), donner le dossier, puis activer la lecture.
            var selecteur = _camera.Parameters[new EnumName("TestImageSelector")];
            bool off = selecteur.TrySetValue("Off");
            bool dossier = _camera.Parameters[new StringName("ImageFilename")].TrySetValue(Path.GetFullPath(DossierImages));
            bool mode = _camera.Parameters[new EnumName("ImageFileMode")].TrySetValue("On");
            Message?.Invoke($"Diagnostic émulateur : TestImageSelector={(selecteur.IsReadable ? selecteur.GetValue() : "absent")} (Off accepté : {off}), " +
                            $"ImageFileMode={_camera.Parameters[new EnumName("ImageFileMode")].GetValueOrDefault("?")}, " +
                            $"ImageFilename={_camera.Parameters[new StringName("ImageFilename")].GetValueOrDefault("?")}, " +
                            $"taille={_camera.Parameters[PLCamera.Width].GetValue()}x{_camera.Parameters[PLCamera.Height].GetValue()}");
            Message?.Invoke(mode && dossier
                ? $"Émulateur : lecture des images de {DossierImages}"
                : $"Émulateur : réglage refusé (ImageFileMode={mode}, ImageFilename={dossier})");
        }

        // Déclenchement logiciel : la caméra ne prend une image que sur commande (BoucleDeclenchement).
        // Sur une vraie ligne, la source serait Line1 : un codeur ou une sortie d'automate.
        _camera.Parameters[PLCamera.TriggerSelector].TrySetValue(PLCamera.TriggerSelector.FrameStart);
        _camera.Parameters[PLCamera.TriggerMode].TrySetValue(PLCamera.TriggerMode.On);
        _camera.Parameters[PLCamera.TriggerSource].TrySetValue(PLCamera.TriggerSource.Software);

        AppliquerExposition();
        _camera.StreamGrabber.Start(GrabStrategy.OneByOne, GrabLoop.ProvidedByStreamGrabber);
        _connectee = true;
    }

    private void AppliquerExposition()
    {
        try
        {
            if (!_camera!.Parameters[PLCamera.ExposureTime].TrySetValue(_exposition))
                _camera.Parameters[PLCamera.ExposureTimeAbs].TrySetValue(_exposition);
        }
        catch (Exception ex)
        {
            Message?.Invoke($"Exposition {_exposition:F0} µs refusée : {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Commandes venant de l'interface
    // ------------------------------------------------------------------
    /// <summary>Cadence d'acquisition en images par seconde (déclenchement logiciel).</summary>
    public void DefinirCadence(double imagesParSeconde) =>
        _periodeMs = (int)Math.Round(1000 / Math.Clamp(imagesParSeconde, 0.5, 60));

    private async Task BoucleDeclenchement(CancellationToken ct)
    {
        try
        {
            while (true)
            {
                lock (_verrou)
                {
                    try
                    {
                        if (_connectee && _camera!.WaitForFrameTriggerReady(100, TimeoutHandling.Return))
                            _camera.ExecuteSoftwareTrigger();
                    }
                    catch (Exception ex) { Message?.Invoke($"Déclenchement impossible : {ex.Message}"); }
                }
                await Task.Delay(_periodeMs, ct);
            }
        }
        catch (OperationCanceledException) { }
    }

    public void DefinirExposition(double microsecondes)
    {
        lock (_verrou)
        {
            _exposition = microsecondes;
            if (_connectee) AppliquerExposition();
        }
    }

    public void ForcerPanne()
    {
        lock (_verrou)
        {
            if (!_connectee) { Message?.Invoke("Impossible : caméra non connectée"); return; }
            _camera!.Parameters[new CommandName("ForceFailedBuffer")].TryExecute();
        }
        Message?.Invoke("Panne d'acquisition forcée");
    }

    public void SimulerDeconnexion()
    {
        Message?.Invoke("Déconnexion simulée");
        _connexionPerdue.Release();
    }

    // ------------------------------------------------------------------
    // PRODUCTEUR : thread pylon. Copier, poser dans la file, rendre la main.
    // ------------------------------------------------------------------
    private void SurImageRecue(object? sender, ImageGrabbedEventArgs e)
    {
        IGrabResult r = e.GrabResult;
        if (r.GrabSucceeded && r.PixelData is byte[] pixels)
        {
            long id = Interlocked.Increment(ref _recues);
            _file?.Writer.TryWrite(new Frame(id, r.Width, r.Height, (byte[])pixels.Clone(), DateTime.UtcNow));
        }
        else
        {
            Interlocked.Increment(ref _ratees);
            Message?.Invoke($"Acquisition ratée : {r.ErrorDescription}");
        }
    }

    // ------------------------------------------------------------------
    // CONSOMMATEUR : l'inférence du modèle, image par image, avec try/catch autour de CHAQUE image
    // ------------------------------------------------------------------
    private async Task BoucleTraitement(ChannelReader<Frame> lecteur)
    {
        await foreach (var frame in lecteur.ReadAllAsync())
        {
            try
            {
                double moyenne = Moyenne(frame.Pixels);
                var detections = new List<Detection>();
                var chrono = Stopwatch.StartNew();

                if (_detecteur is not null)
                    detections = _detecteur.Detecter(frame.Pixels, frame.Largeur, frame.Hauteur);
                else if (DureeTraitementMs > 0)
                    Thread.Sleep(DureeTraitementMs);                    // pas de modèle : traitement simulé

                chrono.Stop();
                Interlocked.Increment(ref _traitees);
                if (detections.Count > 0)
                {
                    Interlocked.Increment(ref _avecDefaut);
                    _journal?.Ecrire(frame.Id, detections, _detecteur?.Version ?? "", chrono.Elapsed.TotalMilliseconds);
                    // Traçabilité : chaque décision est journalisée avec la version du modèle
                    Message?.Invoke($"Image {frame.Id} : {string.Join(", ", detections.Select(d => $"{d.Classe} {d.Score:F2}"))}" +
                                    $"  [{_detecteur?.Version}]");
                }

                ImageTraitee?.Invoke(new ResultatImage(frame.Id, frame.Largeur, frame.Hauteur, frame.Pixels, moyenne,
                                                       DateTime.UtcNow - frame.Horodatage, detections,
                                                       chrono.Elapsed.TotalMilliseconds));
            }
            catch (Exception ex)
            {
                Interlocked.Increment(ref _erreurs);
                Message?.Invoke($"Erreur de traitement sur l'image {frame.Id} : {ex.Message}");
            }
        }
    }

    private void ChargerSeuils(string chemin)
    {
        if (_detecteur is null || !File.Exists(chemin)) return;
        var seuils = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, float>>(File.ReadAllText(chemin));
        foreach (var (classe, seuil) in seuils ?? new())
            _detecteur.SeuilsParClasse[classe] = seuil;
        Message?.Invoke($"Seuils par classe : {string.Join(", ", _detecteur.SeuilsParClasse.Select(kv => $"{kv.Key} {kv.Value:F2}"))}");
    }

    private static double Moyenne(byte[] pixels)
    {
        if (pixels.Length == 0) return 0;
        long somme = 0;
        foreach (byte p in pixels) somme += p;
        return (double)somme / pixels.Length;
    }

    // ------------------------------------------------------------------
    // ROBUSTESSE : déconnexion → on signale, la surveillance reconnecte
    // ------------------------------------------------------------------
    private void SurConnexionPerdue(object? sender, EventArgs e)
    {
        Message?.Invoke("Connexion caméra perdue");
        _connexionPerdue.Release();
    }

    private async Task BoucleSurveillance(CancellationToken ct)
    {
        try
        {
            while (true)
            {
                await _connexionPerdue.WaitAsync(ct);
                await Reconnecter(ct);
            }
        }
        catch (OperationCanceledException) { }
    }

    private async Task Reconnecter(CancellationToken ct)
    {
        lock (_verrou)
        {
            _connectee = false;
            try { _camera?.StreamGrabber.Stop(); } catch { }
            try { _camera?.Close(); } catch { }
        }
        EtatChange?.Invoke(EtatCamera.Degrade);
        Message?.Invoke("Mode dégradé : inspection arrêtée");

        int delai = 1;
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                lock (_verrou) { OuvrirEtDemarrer(); }
                EtatChange?.Invoke(EtatCamera.Acquisition);
                Message?.Invoke("Caméra reconnectée, inspection reprise");
                return;
            }
            catch (Exception ex)
            {
                lock (_verrou) { try { _camera?.Close(); } catch { } }
                Message?.Invoke($"Reconnexion impossible ({ex.Message}), nouvel essai dans {delai} s");
                await Task.Delay(TimeSpan.FromSeconds(delai), ct);
                delai = Math.Min(delai * 2, 30);
            }
        }
    }

    // ------------------------------------------------------------------
    // Arrêt propre
    // ------------------------------------------------------------------
    public async Task ArreterAsync()
    {
        if (_camera is null) return;

        _cts?.Cancel();
        if (_surveillance is not null) await _surveillance;
        if (_declenchement is not null) await _declenchement;

        lock (_verrou)
        {
            _connectee = false;
            try { _camera.StreamGrabber.Stop(); } catch { }
            try { _camera.Close(); } catch { }
        }

        _file?.Writer.TryComplete();
        if (_traitement is not null) await _traitement;
        _journal?.Dispose();                // après la fin du consommateur : plus personne n'écrit
        _journal = null;

        _camera.StreamGrabber.ImageGrabbed -= SurImageRecue;
        _camera.ConnectionLost -= SurConnexionPerdue;
        _camera.Dispose();
        _camera = null;
        _cts?.Dispose();
        _cts = null;

        var s = Statistiques();
        Message?.Invoke($"Arrêt. Bilan : reçues {s.Recues}, traitées {s.Traitees}, avec défaut {s.AvecDefaut}, " +
                        $"perdues {s.Perdues}, ratées {s.Ratees}, erreurs {s.Erreurs}");
        EtatChange?.Invoke(EtatCamera.Arretee);
    }

    public void Dispose()
    {
        _cts?.Cancel();
        _camera?.Dispose();
        _camera = null;
        _detecteur?.Dispose();
        _detecteur = null;
        _journal?.Dispose();
        _journal = null;
    }
}
