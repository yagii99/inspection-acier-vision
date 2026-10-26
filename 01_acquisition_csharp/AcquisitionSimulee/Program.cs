// Chaîne d'acquisition avec une caméra émulée Basler pylon
// - déclenchement logiciel toutes les 50 ms
// - file bornée producteur / consommateur (Channel)
// - traitement volontairement lent (80 ms) pour observer les images perdues
// - pannes forcées avec ForceFailedBuffer pour tester la robustesse
//
// Prérequis : pylon installé avec le support de l'émulation, et PYLON_CAMEMU=1 (ou plus)

using System.Diagnostics;
using System.Threading.Channels;
using Basler.Pylon;

// ---------- Paramètres de la simulation ----------
const int PeriodeDeclenchementMs = 50;   // la "ligne" produit une image toutes les 50 ms
const int DureeTraitementMs = 80;        // le traitement est plus lent que la ligne
const int TailleFile = 5;
const int NbImagesTotal = 300;
const int PanneToutesLesNImages = 60;    // une panne forcée toutes les 60 images
string? dossierImages = args.Length > 0 ? args[0] : null;   // ex. un dossier d'images NEU

// ---------- Compteurs (modifiés depuis plusieurs threads) ----------
long recues = 0, echecs = 0, perdues = 0, traitees = 0;

// ---------- File bornée : on jette la plus vieille image si le traitement ne suit pas ----------
var file = Channel.CreateBounded<Frame>(
    new BoundedChannelOptions(TailleFile)
    {
        FullMode = BoundedChannelFullMode.DropOldest,
        SingleReader = true,
        SingleWriter = true
    },
    _ => Interlocked.Increment(ref perdues));

// ---------- Ouverture de la caméra émulée ----------
var cameras = CameraFinder.Enumerate();
var infoEmu = cameras.FirstOrDefault(c => c[CameraInfoKey.DeviceType] == DeviceType.CamEmu);
if (infoEmu is null)
{
    Console.WriteLine("Aucune caméra émulée trouvée. Vérifie que PYLON_CAMEMU est défini (ex. PYLON_CAMEMU=1) " +
                      "et que le support de l'émulation est installé.");
    return 1;
}

using var camera = new Camera(infoEmu);
camera.CameraOpened += Configuration.SoftwareTrigger;   // TriggerMode = On, TriggerSource = Software
camera.ConnectionLost += (_, _) => Console.WriteLine("!! Connexion caméra perdue");
camera.Open();

Console.WriteLine($"Caméra : {camera.CameraInfo[CameraInfoKey.FriendlyName]}");

// Réglages : on utilise TrySetValue car les noms varient selon les modèles et versions de pylon
camera.Parameters[PLCamera.PixelFormat].TrySetValue(PLCamera.PixelFormat.Mono8);
if (!camera.Parameters[PLCamera.ExposureTime].TrySetValue(2000))
    camera.Parameters[PLCamera.ExposureTimeAbs].TrySetValue(2000);

// Fonctions propres à l'émulateur : lire des images depuis un dossier
if (dossierImages is not null && Directory.Exists(dossierImages))
{
    camera.Parameters[new EnumName("ImageFileMode")].TrySetValue("On");
    camera.Parameters[new StringName("ImageFilename")].TrySetValue(Path.GetFullPath(dossierImages));
    Console.WriteLine($"Images lues depuis : {dossierImages}");
}
else
{
    camera.Parameters[new EnumName("TestImageSelector")].TrySetValue("Testimage1");
    Console.WriteLine("Images de test de l'émulateur (passer un dossier en argument pour utiliser d'autres images)");
}

// ---------- Callback d'acquisition : il copie l'image et rend la main tout de suite ----------
camera.StreamGrabber.ImageGrabbed += (_, e) =>
{
    IGrabResult r = e.GrabResult;
    if (r.GrabSucceeded && r.PixelData is byte[] pixels)
    {
        long id = Interlocked.Increment(ref recues);
        var frame = new Frame(id, r.Width, r.Height, (byte[])pixels.Clone(), DateTime.UtcNow);
        file.Writer.TryWrite(frame);   // ne bloque jamais grâce à DropOldest
    }
    else
    {
        Interlocked.Increment(ref echecs);
        Console.WriteLine($"   Acquisition ratée : {r.ErrorCode:X} {r.ErrorDescription}");
    }
};

// ---------- Consommateur : traitement dans sa propre tâche ----------
using var cts = new CancellationTokenSource();
var traitement = Task.Run(async () =>
{
    await foreach (var frame in file.Reader.ReadAllAsync(cts.Token))
    {
        double moyenne = Traiter(frame);
        long n = Interlocked.Increment(ref traitees);
        if (n % 20 == 0)
            Console.WriteLine($"Image #{frame.Id} traitée, niveau moyen {moyenne:F1}, " +
                              $"retard {(DateTime.UtcNow - frame.Horodatage).TotalMilliseconds:F0} ms");
    }
});

double Traiter(Frame f)
{
    // Stand-in du vrai traitement (détection de défauts) : moyenne des niveaux de gris
    long somme = 0;
    foreach (byte p in f.Pixels) somme += p;
    Thread.Sleep(DureeTraitementMs);
    return (double)somme / f.Pixels.Length;
}

// ---------- Producteur : la "ligne de production" déclenche la caméra ----------
camera.StreamGrabber.Start(GrabStrategy.OneByOne, GrabLoop.ProvidedByStreamGrabber);
var chrono = Stopwatch.StartNew();

try
{
    for (int i = 1; i <= NbImagesTotal; i++)
    {
        if (i % PanneToutesLesNImages == 0)
        {
            // Panne simulée : le prochain buffer sera marqué comme raté
            camera.Parameters[new CommandName("ForceFailedBuffer")].TryExecute();
            Console.WriteLine($"-> Panne forcée avant l'image {i}");
        }

        if (camera.WaitForFrameTriggerReady(1000, TimeoutHandling.Return))
            camera.ExecuteSoftwareTrigger();
        else
            Console.WriteLine("   Caméra pas prête pour le déclenchement");

        Thread.Sleep(PeriodeDeclenchementMs);
    }
}
finally
{
    // Arrêt propre : on laisse le temps aux dernières images d'arriver, puis on ferme tout
    Thread.Sleep(500);
    camera.StreamGrabber.Stop();
    file.Writer.Complete();
    await traitement;
    camera.Close();
}

// ---------- Bilan ----------
Console.WriteLine();
Console.WriteLine($"Durée            : {chrono.Elapsed.TotalSeconds:F1} s");
Console.WriteLine($"Déclenchements   : {NbImagesTotal}");
Console.WriteLine($"Images reçues    : {recues}");
Console.WriteLine($"Acquisitions KO  : {echecs}");
Console.WriteLine($"Images perdues   : {perdues} (file pleine)");
Console.WriteLine($"Images traitées  : {traitees}");
return 0;

record Frame(long Id, int Largeur, int Hauteur, byte[] Pixels, DateTime Horodatage);
