// Robustesse : pannes d'acquisition, erreurs de traitement,
// fermeture propre et reconnexion automatique avec délai croissant.
//
// Touches :
//   P = provoquer une acquisition ratée (commande spéciale de l'émulateur)
//   D = simuler une déconnexion de la caméra (reconnexion au 1er essai)
//   E = simuler une déconnexion où la caméra met du temps à revenir (3 essais ratés)
//   Q = quitter

using System.Threading.Channels;
using Basler.Pylon;

// ---------- Compteurs (modifiés depuis plusieurs threads) ----------
int perdues = 0, ratees = 0, erreursTraitement = 0;

// ---------- Simulation : nombre d'essais de reconnexion qui vont échouer exprès ----------
int echecsASimuler = 0;

// ---------- Réflexe 1 : vérifier qu'une caméra est disponible ----------
var cameras = CameraFinder.Enumerate();
if (cameras.Count == 0)
{
    Console.WriteLine("Aucune caméra trouvée. Vérifie le branchement (ou PYLON_CAMEMU pour l'émulateur).");
    return;
}

// ---------- File bornée producteur / consommateur ----------
var file = Channel.CreateBounded<byte[]>(
    new BoundedChannelOptions(5) { FullMode = BoundedChannelFullMode.DropOldest },
    _ => Interlocked.Increment(ref perdues));

using var camera = new Camera(cameras[0]);
camera.CameraOpened += Configuration.AcquireContinuous;

// Le verrou protège les opérations sur la caméra (ouvrir, fermer, régler)
// qui peuvent venir de deux threads : le clavier et la reconnexion.
var verrou = new object();
bool connectee = false;

using var arret = new CancellationTokenSource();
var connexionPerdue = new SemaphoreSlim(0);   // le "signal" de déconnexion

// ---------- Réflexe 2 : le callback compte les images ratées et continue ----------
camera.StreamGrabber.ImageGrabbed += (sender, e) =>
{
    IGrabResult image = e.GrabResult;
    if (image.GrabSucceeded && image.PixelData is byte[] pixels)
        file.Writer.TryWrite((byte[])pixels.Clone());
    else
    {
        Interlocked.Increment(ref ratees);
        Console.WriteLine($"   !! Acquisition ratée : {image.ErrorDescription}");
    }
};

// ---------- Réflexe 5 : en cas de déconnexion, on signale seulement ----------
camera.ConnectionLost += (sender, e) =>
{
    Console.WriteLine("!! Connexion caméra perdue");
    connexionPerdue.Release();
};

// ---------- Réflexe 3 : try/catch autour de chaque image ----------
var traitement = Task.Run(async () =>
{
    int traitees = 0;
    await foreach (var pixels in file.Reader.ReadAllAsync())
    {
        try
        {
            traitees++;
            if (traitees % 30 == 0)
                throw new Exception("image illisible");   // panne simulée du traitement
            Thread.Sleep(100);
            if (traitees % 10 == 0)
                Console.WriteLine($"Image n°{traitees} OK | perdues {perdues} | ratées {ratees} | erreurs {erreursTraitement}");
        }
        catch (Exception ex)
        {
            Interlocked.Increment(ref erreursTraitement);
            Console.WriteLine($"   !! Erreur de traitement sur l'image n°{traitees} : {ex.Message}");
        }
    }
});

// ---------- Tâche de surveillance : attend le signal, puis reconnecte ----------
var surveillance = Task.Run(async () =>
{
    try
    {
        while (true)
        {
            await connexionPerdue.WaitAsync(arret.Token);
            await Reconnecter(arret.Token);
        }
    }
    catch (OperationCanceledException)
    {
        // arrêt demandé : on sort normalement
    }
});

async Task Reconnecter(CancellationToken ct)
{
    // On passe en mode dégradé : on arrête et on ferme ce qui reste
    lock (verrou)
    {
        connectee = false;
        try { camera.StreamGrabber.Stop(); } catch { }
        try { camera.Close(); } catch { }
    }
    Console.WriteLine("   Inspection arrêtée (mode dégradé)");

    int delai = 1;
    while (!ct.IsCancellationRequested)
    {
        try
        {
            // Simulation : la caméra "n'est pas encore revenue"
            if (Volatile.Read(ref echecsASimuler) > 0)
            {
                Interlocked.Decrement(ref echecsASimuler);
                throw new Exception("caméra toujours absente (simulation)");
            }

            lock (verrou)
            {
                camera.Open();
                camera.StreamGrabber.Start(GrabStrategy.OneByOne, GrabLoop.ProvidedByStreamGrabber);
                connectee = true;
            }
            Console.WriteLine(">> Caméra reconnectée, inspection reprise");
            return;
        }
        catch (Exception ex)
        {
            // Si l'ouverture a réussi mais pas le démarrage, on referme pour repartir propre
            lock (verrou)
            {
                try { camera.Close(); } catch { }
            }

            Console.WriteLine($"   Reconnexion impossible ({ex.Message}), nouvel essai dans {delai} s");
            try
            {
                await Task.Delay(TimeSpan.FromSeconds(delai), ct);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            delai = Math.Min(delai * 2, 30);   // 1, 2, 4, 8, 16, 30, 30...
        }
    }
}

// ---------- Réflexe 4 : try/finally pour toujours fermer proprement ----------
try
{
    lock (verrou)
    {
        camera.Open();
        camera.StreamGrabber.Start(GrabStrategy.OneByOne, GrabLoop.ProvidedByStreamGrabber);
        connectee = true;
    }

    Console.WriteLine("P = panne d'acquisition | D = déconnexion | E = déconnexion longue | Q = quitter");

    while (true)
    {
        var touche = Console.ReadKey(true).Key;
        if (touche == ConsoleKey.Q) break;

        switch (touche)
        {
            case ConsoleKey.P:
                lock (verrou)
                {
                    if (connectee)
                    {
                        camera.Parameters[new CommandName("ForceFailedBuffer")].TryExecute();
                        Console.WriteLine(">> Panne d'acquisition forcée");
                    }
                    else
                        Console.WriteLine("   Impossible : caméra non connectée");
                }
                break;

            case ConsoleKey.D:
                Console.WriteLine(">> Déconnexion simulée");
                connexionPerdue.Release();
                break;

            case ConsoleKey.E:
                Interlocked.Exchange(ref echecsASimuler, 3);
                Console.WriteLine(">> Déconnexion simulée, la caméra reviendra au 4e essai");
                connexionPerdue.Release();
                break;
        }
    }
}
finally
{
    // Ordre d'arrêt : surveillance, caméra, file, traitement
    arret.Cancel();
    await surveillance;

    lock (verrou)
    {
        try { camera.StreamGrabber.Stop(); } catch { }
        try { camera.Close(); } catch { }
    }

    file.Writer.Complete();
    await traitement;

    Console.WriteLine();
    Console.WriteLine($"Bilan : perdues {perdues} | ratées {ratees} | erreurs de traitement {erreursTraitement}");
    Console.WriteLine("Caméra fermée proprement.");
}
