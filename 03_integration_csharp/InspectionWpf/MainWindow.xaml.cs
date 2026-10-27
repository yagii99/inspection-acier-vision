// MainWindow : l'interface opérateur. Tout ce fichier s'exécute sur le THREAD UI.
//
// Affichage de la dernière image avec les boîtes détectées, compteurs (inférence, latence,
// images avec défaut) et journal. Rafraîchissement limité à ~15 i/s par un DispatcherTimer.

using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using OpenCvSharp;

namespace InspectionWpf;

public partial class MainWindow : System.Windows.Window
{
    // ---------- Chemins ----------
    // Par défaut : le dossier "donnees" du projet (copié à côté de l'exécutable à la compilation)
    //   donnees\best.onnx : le modèle
    //   donnees\images\   : les images que la caméra émulée "filme"
    // Les variables d'environnement INSPECTION_MODELE et INSPECTION_IMAGES permettent de pointer ailleurs.
    private static readonly string Donnees = System.IO.Path.Combine(AppContext.BaseDirectory, "donnees");
    private static readonly string CheminModele =
        Environment.GetEnvironmentVariable("INSPECTION_MODELE") ?? System.IO.Path.Combine(Donnees, "best.onnx");
    private static readonly string DossierImages =
        Environment.GetEnvironmentVariable("INSPECTION_IMAGES") ?? System.IO.Path.Combine(Donnees, "images");

    private readonly CameraService _service = new() { CheminModele = CheminModele, DossierImages = DossierImages };
    private readonly DispatcherTimer _timerAffichage;
    private ResultatImage? _dernierResultat;          // écrit par le traitement, lu par le thread UI
    private WriteableBitmap? _bitmap;
    private double _derniereLatenceMs, _derniereMoyenne, _derniereInferenceMs;
    private string _derniersDefauts = "";

    public MainWindow()
    {
        InitializeComponent();
        _service.ImageTraitee += r => Interlocked.Exchange(ref _dernierResultat, r);
        _service.Message += m => Dispatcher.InvokeAsync(() => AjouterJournal(m));
        _service.EtatChange += e => Dispatcher.InvokeAsync(() => AfficherEtat(e));

        _timerAffichage = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(66) };
        _timerAffichage.Tick += (_, _) => RafraichirAffichage();
        RafraichirAffichage();
    }

    private async void Demarrer_Click(object sender, RoutedEventArgs e)
    {
        BoutonDemarrer.IsEnabled = false;
        AfficherEtat(EtatCamera.Connexion);
        try
        {
            await Task.Run(() => _service.Demarrer());   // chargement du modèle + ouverture caméra : hors du thread UI
            BoutonArreter.IsEnabled = true;
            BoutonPanne.IsEnabled = true;
            BoutonDeconnexion.IsEnabled = true;
            _timerAffichage.Start();
        }
        catch (Exception ex)
        {
            AjouterJournal($"Démarrage impossible : {ex.Message}");
            AfficherEtat(EtatCamera.Arretee);
            BoutonDemarrer.IsEnabled = true;
        }
    }

    private async void Arreter_Click(object sender, RoutedEventArgs e)
    {
        BoutonArreter.IsEnabled = false;
        BoutonPanne.IsEnabled = false;
        BoutonDeconnexion.IsEnabled = false;
        await Task.Run(() => _service.ArreterAsync());
        _timerAffichage.Stop();
        RafraichirAffichage();
        BoutonDemarrer.IsEnabled = true;
    }

    private void Panne_Click(object sender, RoutedEventArgs e) => _service.ForcerPanne();
    private void Deconnexion_Click(object sender, RoutedEventArgs e) => _service.SimulerDeconnexion();
    private void Cadence_Changed(object sender, RoutedPropertyChangedEventArgs<double> e) => _service.DefinirCadence(e.NewValue);
    private void Exposition_Changed(object sender, RoutedPropertyChangedEventArgs<double> e) => _service.DefinirExposition(e.NewValue);
    private void Traitement_Changed(object sender, RoutedPropertyChangedEventArgs<double> e) => _service.DureeTraitementMs = (int)e.NewValue;

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        _timerAffichage.Stop();
        Task.Run(() => _service.ArreterAsync()).GetAwaiter().GetResult();
        _service.Dispose();
    }

    // ------------------------------------------------------------------
    // Affichage (thread UI) : image en couleur + boîtes des défauts
    // ------------------------------------------------------------------
    private void RafraichirAffichage()
    {
        var r = Interlocked.Exchange(ref _dernierResultat, null);
        if (r is not null && r.Pixels.Length == r.Largeur * r.Hauteur)
        {
            using var gris = new Mat(r.Hauteur, r.Largeur, MatType.CV_8UC1);
            Marshal.Copy(r.Pixels, 0, gris.Data, r.Pixels.Length);
            using var couleur = new Mat();
            Cv2.CvtColor(gris, couleur, ColorConversionCodes.GRAY2BGR);
            foreach (var d in r.Detections)
            {
                Cv2.Rectangle(couleur, d.Boite, new Scalar(0, 0, 255), 2);
                Cv2.PutText(couleur, $"{d.Classe} {d.Score:F2}", new OpenCvSharp.Point(d.Boite.X, Math.Max(d.Boite.Y - 4, 12)),
                            HersheyFonts.HersheySimplex, 0.4, new Scalar(0, 0, 255), 1);
            }

            if (_bitmap is null || _bitmap.PixelWidth != r.Largeur || _bitmap.PixelHeight != r.Hauteur)
            {
                _bitmap = new WriteableBitmap(r.Largeur, r.Hauteur, 96, 96, PixelFormats.Bgr24, null);
                ImageLive.Source = _bitmap;
            }
            int pas = (int)couleur.Step();
            _bitmap.WritePixels(new Int32Rect(0, 0, r.Largeur, r.Hauteur), couleur.Data, pas * r.Hauteur, pas);

            _derniereLatenceMs = r.Latence.TotalMilliseconds;
            _derniereMoyenne = r.Moyenne;
            _derniereInferenceMs = r.InferenceMs;
            _derniersDefauts = r.Detections.Count == 0 ? "aucun" : string.Join(", ", r.Detections.Select(d => d.Classe).Distinct());
        }

        var s = _service.Statistiques();
        TexteCompteurs.Text =
            $"Reçues          {s.Recues}\n" +
            $"Traitées        {s.Traitees}\n" +
            $"Avec défaut     {s.AvecDefaut}\n" +
            $"Non inspectées  {s.NonInspectees}\n" +
            $"Ratées          {s.Ratees}\n" +
            $"Erreurs         {s.Erreurs}\n" +
            $"Inférence       {_derniereInferenceMs:F1} ms\n" +
            $"Latence         {_derniereLatenceMs:F0} ms\n" +
            $"Luminosité      {_derniereMoyenne:F0}\n" +
            $"Défauts         {_derniersDefauts}";
    }

    private void AjouterJournal(string message)
    {
        ListeJournal.Items.Insert(0, $"{DateTime.Now:HH:mm:ss}  {message}");
        if (ListeJournal.Items.Count > 200)
            ListeJournal.Items.RemoveAt(ListeJournal.Items.Count - 1);
    }

    private void AfficherEtat(EtatCamera etat)
    {
        (VoyantEtat.Fill, TexteEtat.Text) = etat switch
        {
            EtatCamera.Acquisition => (Brushes.LimeGreen, "Acquisition en cours"),
            EtatCamera.Connexion => (Brushes.Orange, "Connexion..."),
            EtatCamera.Degrade => (Brushes.Red, "Mode dégradé : caméra perdue"),
            _ => (Brushes.Gray, "Arrêtée")
        };
    }
}
