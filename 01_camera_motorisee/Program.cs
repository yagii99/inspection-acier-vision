// Séquence d'acquisition avec une caméra motorisée
// Pour chaque position : Déplacement → Stabilisation → Capture → Contrôle → Sauvegarde
// Le moteur est simulé (il se bloque parfois exprès), la caméra est l'émulateur pylon.
// Les images et un journal CSV sont enregistrés dans un dossier "inspection_<date>".

using Basler.Pylon;

// ---------- Plan d'inspection : les positions où la caméra doit regarder ----------
double[] plan = { 0, 30, 60, 90, 120 };   // angles en degrés
const int StabilisationMs = 300;           // attente après déplacement (vibrations)
const int MaxEssaisMoteur = 2;
const int MaxEssaisCapture = 2;

// ---------- Dossier de sortie et journal (traçabilité) ----------
string dossier = Path.Combine(Environment.CurrentDirectory, $"inspection_{DateTime.Now:yyyyMMdd_HHmmss}");
Directory.CreateDirectory(dossier);
string journal = Path.Combine(dossier, "journal.csv");
File.WriteAllText(journal, "horodatage;position_deg;etape;resultat;fichier\n");

// ---------- Caméra ----------
var cameras = CameraFinder.Enumerate();
if (cameras.Count == 0)
{
    Console.WriteLine("Aucune caméra trouvée. Vérifie PYLON_CAMEMU pour l'émulateur.");
    return;
}

var moteur = new MoteurSimule();
var hasard = new Random();

using var camera = new Camera(cameras[0]);
camera.CameraOpened += Configuration.SoftwareTrigger;   // c'est le programme qui décide du moment
camera.Open();
camera.StreamGrabber.Start();   // sans argument : c'est nous qui récupérons chaque image (RetrieveResult)

Console.WriteLine($"Inspection de {plan.Length} positions. Résultats dans : {dossier}\n");

int reussies = 0;
try
{
    foreach (double position in plan)
    {
        bool ok = await InspecterPosition(position);
        if (ok) reussies++;
        Console.WriteLine();
    }

    Console.WriteLine("Retour en position de repos...");
    try { await moteur.AllerAAsync(0); } catch (Exception ex) { Console.WriteLine($"   !! {ex.Message}"); }
}
finally
{
    camera.StreamGrabber.Stop();
    camera.Close();
}

Console.WriteLine($"\nBilan : {reussies} / {plan.Length} positions inspectées. Journal : {journal}");


// ======================================================================
// La machine à états pour UNE position
// ======================================================================
async Task<bool> InspecterPosition(double position)
{
    Etape etape = Etape.Deplacement;
    int essaisMoteur = 0, essaisCapture = 0;
    IGrabResult? image = null;

    while (true)
    {
        Console.WriteLine($"[{position,5}°] {etape}");

        switch (etape)
        {
            case Etape.Deplacement:
                try
                {
                    await moteur.AllerAAsync(position);
                    etape = Etape.Stabilisation;
                }
                catch (Exception ex)
                {
                    essaisMoteur++;
                    Console.WriteLine($"          !! {ex.Message} (essai {essaisMoteur}/{MaxEssaisMoteur})");
                    Journaliser(position, etape, ex.Message, "");
                    if (essaisMoteur >= MaxEssaisMoteur) etape = Etape.Erreur;
                }
                break;

            case Etape.Stabilisation:
                await Task.Delay(StabilisationMs);
                etape = Etape.Capture;
                break;

            case Etape.Capture:
                try
                {
                    if (!camera.WaitForFrameTriggerReady(1000, TimeoutHandling.Return))
                        throw new Exception("caméra pas prête pour le déclenchement");
                    camera.ExecuteSoftwareTrigger();
                    image = camera.StreamGrabber.RetrieveResult(5000, TimeoutHandling.ThrowException);
                    etape = Etape.Controle;
                }
                catch (Exception ex)
                {
                    essaisCapture++;
                    Console.WriteLine($"          !! {ex.Message}");
                    Journaliser(position, etape, ex.Message, "");
                    etape = essaisCapture >= MaxEssaisCapture ? Etape.Erreur : Etape.Capture;
                }
                break;

            case Etape.Controle:
                var (bonne, raison) = ControlerQualite(image!);
                if (bonne)
                {
                    Console.WriteLine($"          image OK ({raison})");
                    etape = Etape.Sauvegarde;
                }
                else
                {
                    essaisCapture++;
                    Console.WriteLine($"          !! image refusée : {raison} (essai {essaisCapture}/{MaxEssaisCapture})");
                    Journaliser(position, etape, raison, "");
                    image!.Dispose();
                    image = null;
                    etape = essaisCapture >= MaxEssaisCapture ? Etape.Erreur : Etape.Capture;
                }
                break;

            case Etape.Sauvegarde:
                string fichier = Path.Combine(dossier, $"pos_{position:000}.png");
                ImagePersistence.Save(ImageFileFormat.Png, fichier, image!);
                Journaliser(position, etape, "OK", fichier);
                image!.Dispose();
                image = null;
                etape = Etape.Termine;
                break;

            case Etape.Termine:
                Console.WriteLine($"          position {position}° inspectée");
                return true;

            case Etape.Erreur:
                image?.Dispose();
                Console.WriteLine($"          position {position}° ABANDONNÉE, on passe à la suivante");
                Journaliser(position, etape, "abandon", "");
                return false;
        }
    }
}

// Contrôle simple : acquisition réussie, image ni trop sombre ni trop claire.
// Plus une "image floue" simulée de temps en temps pour tester la recapture.
(bool, string) ControlerQualite(IGrabResult img)
{
    if (!img.GrabSucceeded)
        return (false, $"acquisition ratée : {img.ErrorDescription}");
    if (img.PixelData is not byte[] pixels || pixels.Length == 0)
        return (false, "image vide");

    double moyenne = pixels.Average(p => (double)p);
    if (moyenne < 20) return (false, $"trop sombre (moyenne {moyenne:F0})");
    if (moyenne > 235) return (false, $"trop claire (moyenne {moyenne:F0})");
    if (hasard.NextDouble() < 0.2) return (false, "image floue (simulation)");

    return (true, $"luminosité moyenne {moyenne:F0}");
}

void Journaliser(double position, Etape etape, string resultat, string fichier)
{
    File.AppendAllText(journal, $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff};{position};{etape};{resultat};{fichier}\n");
}


// ======================================================================
// Types
// ======================================================================
enum Etape { Deplacement, Stabilisation, Capture, Controle, Sauvegarde, Termine, Erreur }

// Simule un moteur pan : il met un temps proportionnel à la distance,
// et se bloque parfois (15 % des déplacements) pour tester la gestion d'erreur.
class MoteurSimule
{
    private readonly Random _hasard = new();
    private const double VitesseDegParSeconde = 60;

    public double Position { get; private set; }

    public async Task AllerAAsync(double cible)
    {
        if (_hasard.NextDouble() < 0.15)
        {
            await Task.Delay(200);
            throw new Exception("moteur bloqué (simulation)");
        }

        double distance = Math.Abs(cible - Position);
        await Task.Delay(TimeSpan.FromSeconds(distance / VitesseDegParSeconde));
        Position = cible;
    }
}