# jobgpumonitor : document de conception (v1, avant code)

## 0. Périmètre (décision du 2026-09-03)

**jobgpumonitor est un émetteur, pas un serveur.** Il ne vit que pendant l'exécution, donc il n'expose rien : il produit des messages (événements) et les envoie. Le stockage, l'API, les notifications et les dashboards sont dans un **projet serveur séparé** qui consomme ces messages.

Ce projet contient donc :

1. La **bibliothèque Python in-process** (stdlib-only) qui observe le programme et émet des événements.
2. Le **protocole d'événements** versionné (JSON Schema) : c'est le contrat entre l'émetteur et n'importe quel serveur.
3. Les **transports** d'envoi : fichier JSONL sur système de fichiers partagé, HTTP vers un serveur, avec tampon disque hors-ligne.
4. Une **CLI minimale** côté émission : `jgm run <cmd>` pour les programmes non Python, `jgm emit` pour envoyer un événement depuis un script bash, `jgm doctor` pour vérifier l'installation.
5. Optionnel, à décider : une **sonde ordonnanceur** (`jgm scheduler-probe`) qui tourne sur le login node et émet dans le même protocole les états squeue/sacct/oarstat. Voir 1.2.

Ce projet ne contient pas : base de données, API REST, notifications ntfy/Telegram, TUI, dashboard. Ils vont dans le projet serveur.

## 1. Le problème et les contraintes

### 1.1 Trois machines

- **Nœud de calcul** : là où tourne le programme. Souvent sans Internet, souvent dans un conteneur enroot/apptainer où `squeue` n'existe pas et où `$HOME` n'est pas forcément monté (le `launcher.sh` de référence ne monte que PROJECT, DATA, OUTPUT, PIP_CACHE).
- **Nœud de login** : a `squeue`, `sacct`, `oarstat`, voit le FS partagé, a souvent Internet via proxy.
- **Laptop** : joignable seulement en SSH sortant vers le cluster.

Conséquence : l'émetteur ne peut pas supposer de réseau. Il doit pouvoir "envoyer" en écrivant sur le FS partagé, et pousser en HTTP seulement quand un serveur est joignable.

### 1.2 Ce que l'in-process ne peut pas savoir

Après un SIGKILL (OOM, scancel, walltime, panne de nœud), aucun code Python ne s'exécute. Seul l'ordonnanceur sait que l'état final est `OUT_OF_MEMORY`, `TIMEOUT`, `CANCELLED`, `PREEMPTED`, `NODE_FAIL`. Il connaît aussi la phase PENDING et sa raison, que l'in-process ne voit jamais puisqu'il n'est pas encore lancé.

Quelqu'un doit donc interroger l'ordonnanceur depuis le login node. Deux options :

- **A. Le serveur le fait.** Simple, mais oblige le serveur à tourner sur le login node ou à faire du SSH.
- **B. Ce projet fournit une sonde ordonnanceur** qui émet dans le même protocole. Le serveur reste un pur consommateur, déployable n'importe où. Le principe "ce projet émet, le serveur consomme" est respecté à la lettre.

Recommandation : B, en phase 2. Le protocole prévoit dès la phase 1 les événements `scheduler.*` pour ne pas avoir à le casser.

## 2. Le protocole d'événements

C'est le livrable le plus important : tout serveur, dashboard ou script pourra le consommer sans dépendre de notre code.

### 2.1 Enveloppe

Une ligne JSON par événement (JSONL), UTF-8, append-only. Champs communs :

```json
{
  "v": 1,
  "id": "01J8…",                       // ULID, unique, triable dans le temps
  "seq": 42,                           // numéro de séquence par run (détection de perte)
  "ts": "2026-09-03T10:14:05.123Z",    // UTC
  "mono": 1234.56,                     // horloge monotone du processus (durées fiables)
  "run_id": "marcel-c3/1234567/0",     // cluster/job_id/restart_count ; "local/<host>/<pid>/<start>" sans ordonnanceur
  "source": "process",                 // process | scheduler | wrapper
  "rank": 0,                           // rang DDP, null sinon
  "type": "run.start",
  "data": { … }
}
```

### 2.2 Types d'événements (phase 1)

| Type | Quand | Contenu principal |
|---|---|---|
| `run.start` | premier import / début du wrapper | ordonnanceur détecté, job id/name, array ids, restart count, host, user, cwd, argv, git commit + dirty, python, versions de packages clés, env filtré, chemins stdout/stderr réels via `/proc/self/fd/1` et `fd/2`, deadline si `SLURM_JOB_END_TIME`, ressources visibles (GPUs, cpus, limite mémoire cgroup) |
| `run.heartbeat` | toutes les 15 s (configurable) | uptime, dernier `seq`, résumé ressources compact |
| `resource.sample` | toutes les 10 à 60 s | par GPU : util, mem used/total, temp, power ; CPU %, RSS, mémoire cgroup used/max, disque du dossier de sortie |
| `progress.update` | à chaque tick tqdm (limité en fréquence) | desc, n, total, rate, eta_s, `eta_vs_deadline` |
| `metric.log` | `log()` ou callback framework | dict de scalaires, step, epoch |
| `log.line` | handler `logging` niveau ≥ WARNING, et regex stdout optionnelle | logger, level, message |
| `signal.received` | SIGTERM, SIGUSR1, SIGUSR2, SIGINT | signal, temps restant estimé |
| `run.exception` | excepthook / threading.excepthook | type, message, traceback, dernières frames avec locals optionnels et masqués |
| `run.end` | atexit ou fin du wrapper | exit_code, status (`ok` / `error` / `interrupted`), durée, dernières métriques, résumé d'efficacité |
| `checkpoint.saved` | nouveaux fichiers dans le dossier surveillé | chemin, taille |
| `stack.dump` | sur demande (fichier commande) | piles de toutes les threads |
| `scheduler.state` | sonde login node (phase 2) | state, reason, node, time_limit, start/end estimés, exit code sacct, MaxRSS |

Les champs inconnus sont ignorés par les consommateurs ; ajouter un champ ne change pas `v` ; retirer ou renommer incrémente `v`. Schémas JSON publiés dans `schema/`.

### 2.3 Ce que le serveur peut dériver (et donc ce qu'on ne fait pas ici)

Job figé (plus de heartbeat), GPU inactif, dépassement de walltime prévisible, mémoire proche de la limite, agrégation des arrays, dédup et notifications : ce sont des règles sur le flux, elles appartiennent au serveur. L'émetteur fournit les données brutes nécessaires, à bonne fréquence, et une seule chose de pré-calculée : `eta_vs_deadline` dans `progress.update`, parce que seul le processus a l'ETA de tqdm et la deadline au même endroit.

## 3. Transports

L'émetteur écrit vers une liste de **sinks**, tous à échec silencieux :

| Sink | Usage | Détail |
|---|---|---|
| `file` (défaut) | tout cluster avec FS partagé | `$JGM_DIR/runs/<run_id>/events.jsonl`, `flush` + `fsync` par événement (NFS close-to-open). Repli si `$HOME` n'est pas monté : `$JGM_DIR` explicite, sinon `./.jgm` dans le cwd. |
| `http` | serveur joignable depuis le nœud | `JGM_URL=http://login-01:8765/ingest`, POST en batch (toutes les 5 s ou 100 événements), gzip, token bearer. `urllib` stdlib, pas de dépendance. |
| `spool` | tampon hors-ligne du sink http | file disque locale, rejeu avec backoff, rejeu final à `run.end` ; si toujours injoignable, le fichier reste et le serveur pourra l'importer plus tard. |
| `stderr` | debug | une ligne par événement, pratique dans les .out |

Le serveur peut donc être alimenté soit en lisant `$JGM_DIR` (il tourne alors sur le login node), soit en HTTP (il tourne n'importe où joignable depuis les nœuds). Les deux au besoin.

Exemple d'ajout dans `launcher.sh` :

```bash
--container-mounts="…,${HOME}/.jobgpumonitor:/jgm"
export JGM_DIR=/jgm JGM_AUTO=1 PYTHONUNBUFFERED=1
```

## 4. Niveaux d'intégration

| Niveau | Geste utilisateur | Résultat |
|---|---|---|
| L1 | `export JGM_AUTO=1` dans le script sbatch, ou `jgm run python train.py` | Tout, sans toucher au code, y compris les erreurs d'import (hook `.pth` à la coverage.py, actif seulement si la variable est définie) |
| L2 | `import jobgpumonitor.auto` | Une ligne. Ne capte pas les exceptions levées avant cette ligne. |
| L3 | `jobgpumonitor.log(loss=0.3, epoch=4)` ; callbacks Lightning / HF Trainer / Keras auto-détectés | Métriques nommées |

Limitations du one-liner par rapport à une intégration explicite : aucune, sauf métriques personnalisées (appel `log()`), configuration par environnement, et ordre d'import pour les exceptions précoces, ce que L1 supprime.

Règle d'or : **ne jamais changer le comportement du programme.**

- `sys.excepthook`, `threading.excepthook`, handlers de signaux : chaînés, jamais remplacés. Le SIGTERM de l'utilisateur pour checkpointer continue de fonctionner.
- Aucune exception de notre code ne remonte. Disque plein, FS en panne, serveur injoignable : on jette l'événement.
- Threads `daemon=True`, flush `atexit` borné par un timeout de quelques secondes.
- `os.register_at_fork` : désactivation dans les enfants (workers DataLoader, `multiprocessing`). Garde contre la ré-exécution du module principal en mode `spawn`.
- DDP / torchrun : rang 0 complet, autres rangs en mode allégé (heartbeat + crash), `rank` dans l'enveloppe.
- Coût cible : < 1 % CPU d'un cœur, aucune allocation GPU, aucune synchronisation CUDA (on lit NVML, jamais `torch.cuda`).

## 5. Ce que la bibliothèque capture

- **Identité** : ordonnanceur via env (`SLURM_JOB_ID`, `OAR_JOB_ID`, `PBS_JOBID`, `LSB_JOBID`, sinon `local`), job id, nom, array id/task id, `SLURM_RESTART_COUNT`, hostname, user, cwd, argv, git commit et dirty, Python, versions torch/numpy/transformers, env masqué (KEY, TOKEN, SECRET, PASSWORD, et une liste configurable).
- **Chemins des logs** : `os.readlink("/proc/self/fd/1")` et `fd/2`. Attention, c'est le chemin vu du conteneur, on émet aussi la table des montages si disponible pour que le serveur traduise.
- **Deadline** : `SLURM_JOB_END_TIME` (Slurm ≥ 23.02), sinon `OAR_JOB_WALLTIME_SECONDS` + start, sinon un fichier `deadline` déposé par la sonde ou le serveur dans le dossier du run.
- **Ressources** : NVML si `nvidia-ml-py` est installé (extra `[gpu]`), sinon `nvidia-smi --query-gpu` en sous-processus ; psutil si présent, sinon `/proc/self/status` et `/proc/stat` ; cgroup v2 `memory.current` et `memory.max`, v1 en repli ; MIG et UUID dans `CUDA_VISIBLE_DEVICES` gérés ; ROCm plus tard via une couche "accelerator provider".
- **Progression** : patch de `tqdm.tqdm.update` et `close` (limité à 1 événement/s par barre), handler `logging`, regex stdout optionnelle (`loss=0.12`), callbacks frameworks.
- **Fin de vie** : `faulthandler` activé (un segfault CUDA laisse une trace dans stderr) ; `atexit` ; `os._exit` et SIGKILL ne sont pas capturables, le serveur les déduit du silence + état ordonnanceur.

## 6. Cas limites et réponses

1. **SIGKILL** : rien côté processus. Le protocole garantit heartbeat régulier + `seq` ; le serveur voit l'arrêt et l'état sacct.
2. **Crash avant Python** : rien à émettre. La sonde ordonnanceur ou le serveur lisent le .err. Le hook `.pth` capte au moins les erreurs d'import Python.
3. **Job arrays** : `array_job_id` et `array_task_id` dans `run.start`, un run par tâche. L'agrégation est côté serveur.
4. **Requeue / préemption** : `run_id` inclut `restart_count`, chaque exécution est un run distinct rattaché au même job.
5. **Conteneur sans `$HOME`** : `JGM_DIR` explicite, repli cwd, message clair dans stderr au démarrage si aucun sink n'est utilisable.
6. **NFS / Lustre** : append-only, une ligne par événement, fsync ; un lecteur ne voit jamais un JSON tronqué sinon la dernière ligne, qu'il ignore jusqu'à complétion.
7. **Disque plein, quota** : échec silencieux, compteur d'événements perdus reporté dans le heartbeat suivant.
8. **Interactif, Jupyter** : détecté (`salloc`, `srun --pty`, kernel ipykernel), `run.end` marqué `interactive`.
9. **stdout bufferisé** : on ne touche pas au buffering de l'utilisateur, on documente `PYTHONUNBUFFERED=1`.
10. **Multi-processus** : fork, spawn, DDP, voir section 4. Handle NVML invalide après fork : réinit ou désactivation.
11. **Signal `B:SIGTERM@300`** : avec `B:`, seul le shell batch reçoit le signal, pas le Python du srun. Documenté ; la deadline émise permet au serveur de prévenir quand même.
12. **Serveur HTTP absent** : spool disque, rejeu, jamais de blocage, timeout court sur chaque POST.
13. **Horloges** : UTC + monotone ; le serveur peut recaler avec l'écart entre `ts` de réception et d'émission.
14. **Jobs de 30 jours** : fréquence des `resource.sample` adaptative (10 s la première heure, puis 60 s), rotation du JSONL par taille avec numérotation.
15. **Secrets** : masquage env, locals des frames désactivés par défaut, niveau de détail configurable.
16. **Python ancien** : cible 3.9+, zéro dépendance. Image conteneur avec son propre Python : `pip install jobgpumonitor` y est trivial.
17. **Programme non Python** : `jgm run` capture exit code, durée, ressources, tail de stderr, et émet le même protocole. `jgm emit type key=value` depuis bash.
18. **Notebooks** : `%load_ext jobgpumonitor` plus tard.
19. **Plusieurs instances dans un même processus** : `watch()` idempotent, un seul run par processus.
20. **Windows local** : `/proc` absent, sondes dégradées, tout le reste marche.

## 7. Hors périmètre, à faire dans le projet serveur

Stockage SQLite, API REST + SSE, réconciliation avec l'ordonnanceur, règles d'alerte (GPU inactif, mémoire, walltime, silence), notifications (ntfy, Telegram, Slack, mail, webhook), digest, TUI, dashboard web, explication des raisons d'attente, rapport d'efficacité étendu, multi-utilisateur. Ce projet lui livre le protocole et un jeu de fichiers JSONL d'exemple pour ses tests.

## 8. Packaging et layout

- PyPI `jobgpumonitor`, CLI `jgm`, Python ≥ 3.9, `pyproject.toml`, `uv` pour le build et les tests.
- Zéro dépendance. Extras : `[gpu]` nvidia-ml-py + psutil, `[integrations]` rien à installer (détection dynamique), `[dev]` pytest, ruff, mypy.
- Config par priorité : flags de `watch()` > variables `JGM_*` > `~/.config/jobgpumonitor/config.toml` (tomllib 3.11+, parseur minimal en repli sous 3.11 pour rester sans dépendance).

```
src/jobgpumonitor/
  __init__.py        watch(), log(), finish(), current_run(), emit()
  auto.py            import à effet de bord
  _pth_hook.py       hook JGM_AUTO installé via un fichier .pth
  context.py         détection ordonnanceur, conteneur, rang, git, env masqué
  events.py          dataclasses des événements, sérialisation, ULID, seq
  sinks/             file.py, http.py, spool.py, stderr.py
  probes/            gpu.py (nvml, nvidia-smi), cpu.py, cgroup.py, checkpoints.py
  hooks/             excepthook.py, signals.py, tqdm.py, logging.py, stdout_regex.py, fork.py
  integrations/      lightning.py, transformers.py, keras.py
  cli/               run.py, emit.py, doctor.py, (scheduler_probe.py en phase 2)
schema/              JSON Schema par type d'événement, versionnés
docs/                DESIGN.md, PROTOCOL.md, INTEGRATION.md (launcher.sh, DDP, conteneurs)
tests/               unitaires (contexte, sinks, hooks), intégration (sous-processus qui crashe, OOM simulé, fork), fixtures JSONL
```

## 9. Phases

1. **Fondations** : contexte, événements, sink fichier, hooks exception/signaux/atexit, heartbeat, sondes GPU/CPU/cgroup, tqdm, `auto`, `watch()`, `log()`, `jgm run`, `jgm doctor`, schémas JSON, tests. Livrable : fichiers JSONL exploitables par le futur serveur.
2. **Transport et couverture** : sink HTTP + spool, hook `.pth`, `logging` et regex stdout, DDP, checkpoints, stack dump à la demande, intégrations frameworks, sonde ordonnanceur Slurm puis OAR.
3. **Écosystème** : magic Jupyter, ROCm, PBS/LSF dans la sonde, `jgm emit` enrichi.

## 10. Questions ouvertes

- La sonde ordonnanceur (option B, section 1.2) est-elle dans ce projet ou dans le serveur ?
- Transport par défaut : fichier seul en phase 1, HTTP en phase 2. D'accord ?
- Version de Python sur le login node et dans vos images (cible 3.9).
- Accès à un cluster OAR pour tester ?

## 11. État au 2026-09-03

Phase 1 implémentée et testée (46 tests, macOS, Python 3.12) : bibliothèque stdlib-only,
sink fichier, hooks exception / sys.exit / signaux / tqdm / logging / faulthandler,
sondes GPU (NVML ou nvidia-smi), processus, cgroup, disque, `jgm run|emit|doctor|ls`,
schéma JSON du protocole. Décisions : sonde ordonnanceur dans ce projet en phase 2,
transport HTTP en phase 2, OAR en détection d'environnement seulement (pas de cluster
de test), Python ≥ 3.9. Pas encore validé sur un vrai nœud Slurm avec GPU.
