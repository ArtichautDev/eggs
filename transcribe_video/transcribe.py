#!/usr/bin/env python3
"""
Transcription détaillée d'une vidéo de cours avec l'API OpenAI.

Pipeline :
  1. ffmpeg extrait l'audio (mono 16 kHz, mp3 léger) et le découpe en morceaux
     d'environ 10 min, coupés dans les silences (pas au milieu d'un mot).
  2. Speech to text (gpt-4o-transcribe par défaut) sur chaque morceau.
  3. Un modèle texte pas cher (gpt-4.1-mini par défaut) corrige les erreurs de
     transcription (termes techniques, noms propres, ponctuation, paragraphes)
     SANS résumer ni simplifier. Si des slides (PDF) sont fournies, elles servent
     de glossaire et le modèle marque les passages d'une slide à l'autre.
  4. Écriture d'un fichier Markdown.

Les résultats intermédiaires sont mis en cache dans <sortie>_work/ : si le script
plante ou si tu le relances, il ne repaie pas ce qui est déjà fait.

Usage :
  export OPENAI_API_KEY=sk-...
  python transcribe.py cours.mp4
  python transcribe.py cours.mp4 --slides slides.pdf --language fr
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

MARKER_RE = re.compile(r"\[\[SLIDE\s+(\d+)\]\]")


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def retry(fn, tries: int = 5, what: str = "appel API"):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            wait = 2 ** (i + 1)
            log(f"  {what} a échoué ({type(e).__name__}: {e}), nouvel essai dans {wait}s")
            time.sleep(wait)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def extract_audio(video: Path, out: Path) -> None:
    r = run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "48k", str(out)])
    if r.returncode != 0:
        sys.exit(f"ffmpeg a échoué :\n{r.stderr[-1500:]}")


def audio_duration(path: Path) -> float:
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    return float(r.stdout.strip())


def detect_silences(path: Path) -> list[tuple[float, float]]:
    r = run(["ffmpeg", "-i", str(path), "-af", "silencedetect=noise=-35dB:d=0.4",
             "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(starts, ends))


def plan_cuts(duration: float, silences, target: float, window: float = 90.0) -> list[float]:
    """Points de coupe proches de k*target, placés au milieu du silence le plus proche."""
    mids = [(a + b) / 2 for a, b in silences]
    cuts, t = [], target
    while t < duration - target * 0.25:
        cands = [m for m in mids if abs(m - t) <= window]
        cut = min(cands, key=lambda m: abs(m - t)) if cands else t
        cuts.append(cut)
        t = cut + target
    return cuts


def split_audio(audio: Path, work: Path, target: float) -> list[tuple[Path, float]]:
    duration = audio_duration(audio)
    if duration <= target * 1.25:
        return [(audio, 0.0)]
    cuts = plan_cuts(duration, detect_silences(audio), target)
    bounds = [0.0] + cuts + [duration]
    chunks = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        p = work / f"chunk_{i:03d}.mp3"
        if not p.exists():
            r = run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(audio),
                     "-c:a", "libmp3lame", "-b:a", "48k", str(p)])
            if r.returncode != 0:
                sys.exit(f"ffmpeg a échoué :\n{r.stderr[-1500:]}")
        chunks.append((p, a))
    return chunks


# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #
def load_slides(pdfs: list[Path], client: OpenAI, llm_model: str, work: Path,
                vision: bool) -> list[str]:
    """Texte de chaque slide (numérotation continue sur tous les PDF)."""
    cache = work / "slides.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("PyMuPDF manquant : pip install pymupdf (ou lance sans --slides)")

    slides: list[str] = []
    for pdf in pdfs:
        doc = fitz.open(pdf)
        for page in doc:
            text = page.get_text().strip()
            if len(text) < 30 and vision:
                png = page.get_pixmap(dpi=110).tobytes("png")
                b64 = base64.b64encode(png).decode()
                resp = retry(lambda: client.chat.completions.create(
                    model=llm_model, temperature=0,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Retranscris fidèlement tout le texte de cette slide "
                         "(titre en première ligne). Si elle contient un schéma, un graphique ou une "
                         "formule, décris-le en une ou deux phrases. Pas d'autre commentaire."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]}]), what="lecture slide")
                text = resp.choices[0].message.content.strip()
            slides.append(text)
    empty = sum(1 for s in slides if len(s) < 30)
    if empty:
        log(f"  Attention : {empty}/{len(slides)} slides sans texte exploitable "
            f"(images ?). Relance avec --vision-slides pour les lire par vision.")
    cache.write_text(json.dumps(slides, ensure_ascii=False), encoding="utf-8")
    return slides


def slide_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:90]
    return ""


def slides_context(slides: list[str], current: int) -> str:
    """Sommaire de toutes les slides + texte complet autour de la slide courante."""
    toc = "\n".join(f"{i + 1}. {slide_title(s) or '(sans texte)'}" for i, s in enumerate(slides))
    if len(slides) <= 40:
        lo, hi = 1, len(slides)
    else:
        lo, hi = max(1, current - 2), min(len(slides), current + 10)
    detail = "\n\n".join(f"### Slide {i}\n{slides[i - 1]}" for i in range(lo, hi + 1))
    return f"SOMMAIRE DES SLIDES :\n{toc}\n\nTEXTE DES SLIDES (slide {lo} à {hi}) :\n{detail}"


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_BASE = """Tu es un correcteur de transcriptions de cours enregistrés. Tu reçois la \
transcription brute d'un passage de vidéo (produite par un modèle de reconnaissance vocale) et tu \
renvoies la version corrigée.

RÈGLES STRICTES
- Ne résume JAMAIS, ne raccourcis JAMAIS, ne reformule pas pour simplifier. Chaque idée, exemple, \
digression, chiffre, nom et nuance dite par l'orateur doit rester. Le texte corrigé doit avoir \
quasiment la même longueur que l'entrée.
- Corrige les erreurs de reconnaissance (mots mal entendus, termes techniques, noms propres, \
sigles, formules) en t'aidant du contexte{slides_hint}.
- Corrige la ponctuation et les majuscules, découpe en paragraphes lisibles (une idée par paragraphe).
- Tu peux supprimer les hésitations pures (euh, hum, bégaiements, mots répétés par erreur) et les \
faux départs évidents, rien d'autre. Garde le ton oral de l'orateur.
- Si l'orateur pose une question au public, ou cite une personne, garde-le tel quel.
- N'ajoute aucune information qui n'a pas été dite. Si un passage est incompréhensible, garde-le \
et ajoute [inaudible?] au lieu d'inventer.
- Langue de sortie : la même que celle de l'orateur. Pas de commentaire, pas d'introduction, \
uniquement le texte corrigé en Markdown simple (paragraphes, pas de titres)."""

SLIDES_RULES = """

MARQUAGE DES SLIDES
- Les slides du cours t'ont été données. Dès que l'orateur passe à une autre slide (il commence à \
traiter son contenu, dit "passons à", "ensuite", "sur cette slide", enchaîne sur le sujet suivant...), \
insère sur une ligne seule le marqueur [[SLIDE n]] (n = numéro de la nouvelle slide), juste avant le \
premier paragraphe qui la concerne.
- Le passage commence sur la slide {current}. Ne répète pas son marqueur au début, sauf si le \
passage commence clairement sur une autre slide (dans ce cas, mets le bon marqueur en première ligne).
- Ne marque que ce dont tu es raisonnablement sûr, d'après le contenu de ce qui est dit rapproché du \
texte des slides. L'orateur peut revenir en arrière ou sauter des slides. En cas de doute, ne change pas de slide.
- Les slides ne sont PAS à recopier ni à résumer : seul ce que dit l'orateur est transcrit."""


# --------------------------------------------------------------------------- #
# Étapes API
# --------------------------------------------------------------------------- #
def transcribe_chunk(client: OpenAI, path: Path, model: str, language: str | None,
                     prompt: str) -> str:
    def call():
        with open(path, "rb") as f:
            kwargs = dict(model=model, file=f, response_format="text")
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            r = client.audio.transcriptions.create(**kwargs)
        return r if isinstance(r, str) else r.text
    return retry(call, what="transcription").strip()


def word_count(s: str) -> int:
    return len(s.split())


def correct_chunk(client: OpenAI, model: str, raw: str, previous_tail: str, slides: list[str] | None,
                  current_slide: int, glossary_hint: str, strict: bool = False) -> str:
    system = SYSTEM_BASE.format(slides_hint=" et du texte des slides" if slides else "")
    if slides:
        system += SLIDES_RULES.format(current=current_slide)
    if strict:
        system += ("\n\nATTENTION : ta précédente réponse était beaucoup trop courte, donc tu as résumé. "
                   "Recommence en gardant TOUT le contenu, phrase par phrase.")
    parts = []
    if slides:
        parts.append(slides_context(slides, current_slide))
    elif glossary_hint:
        parts.append(f"CONTEXTE DU COURS : {glossary_hint}")
    if previous_tail:
        parts.append(f"FIN DU PASSAGE PRÉCÉDENT (déjà corrigé, pour la continuité, ne pas le recopier) :\n{previous_tail}")
    parts.append(f"TRANSCRIPTION BRUTE À CORRIGER :\n{raw}")
    resp = retry(lambda: client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": "\n\n".join(parts)}]), what="correction")
    return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #
def render_markers(text: str, slides: list[str] | None) -> str:
    """[[SLIDE n]] -> séparateur Markdown lisible, en supprimant les doublons consécutifs."""
    last = [None]

    def sub(m):
        n = int(m.group(1))
        if n == last[0]:
            return ""
        last[0] = n
        title = slide_title(slides[n - 1]) if slides and 1 <= n <= len(slides) else ""
        label = f"Slide {n}" + (f" : {title}" if title else "")
        return f"\n\n---\n\n### {label}\n\n"

    out = MARKER_RE.sub(sub, text)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcription détaillée d'une vidéo (OpenAI).")
    ap.add_argument("video", type=Path, help="vidéo (ou audio) à transcrire")
    ap.add_argument("--slides", type=Path, nargs="+", help="PDF(s) des slides (optionnel)")
    ap.add_argument("--vision-slides", action="store_true",
                    help="lit par vision les slides sans texte (PDF d'images)")
    ap.add_argument("-o", "--output", type=Path, help="fichier .md de sortie")
    ap.add_argument("--language", help="code langue de l'audio, ex. fr, en (recommandé)")
    ap.add_argument("--stt-model", default="gpt-4o-transcribe",
                    help="modèle speech to text (défaut gpt-4o-transcribe ; whisper-1 en alternative)")
    ap.add_argument("--llm-model", default="gpt-4.1-mini", help="modèle de correction")
    ap.add_argument("--chunk-minutes", type=float, default=10.0, help="durée cible des morceaux")
    ap.add_argument("--no-correct", action="store_true", help="saute l'étape de correction")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Définis OPENAI_API_KEY (export OPENAI_API_KEY=sk-...)")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg est requis (brew install ffmpeg / apt install ffmpeg)")
    if not args.video.exists():
        sys.exit(f"Fichier introuvable : {args.video}")

    out_path = args.output or args.video.with_suffix(".transcription.md")
    work = out_path.with_name(out_path.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # 1. Audio
    audio = work / "audio.mp3"
    if not audio.exists():
        log("1/4 Extraction audio...")
        extract_audio(args.video, audio)
    chunks = split_audio(audio, work, args.chunk_minutes * 60)
    log(f"    {len(chunks)} morceau(x), durée totale {fmt_time(audio_duration(audio))}")

    # 2. Slides
    slides = None
    if args.slides:
        log("2/4 Lecture des slides...")
        slides = load_slides(args.slides, client, args.llm_model, work, args.vision_slides)
        log(f"    {len(slides)} slides")
    glossary = ""
    if slides:
        glossary = "Termes du cours : " + " ; ".join(t for t in (slide_title(s) for s in slides) if t)
        glossary = glossary[:900]

    # 3. Transcription brute
    log("3/4 Speech to text...")
    raws: list[str] = []
    for i, (path, start) in enumerate(chunks):
        cache = work / f"raw_{i:03d}.txt"
        if cache.exists():
            raws.append(cache.read_text(encoding="utf-8"))
            continue
        log(f"    morceau {i + 1}/{len(chunks)} (à partir de {fmt_time(start)})")
        prompt = glossary
        if raws:
            prompt = (prompt + "\n" + raws[-1][-300:]).strip()
        text = transcribe_chunk(client, path, args.stt_model, args.language, prompt)
        cache.write_text(text, encoding="utf-8")
        raws.append(text)

    # 4. Correction
    log("4/4 Correction...")
    corrected: list[str] = []
    current = 1
    tail = ""
    for i, raw in enumerate(raws):
        cache = work / f"corrected_{i:03d}.txt"
        if cache.exists():
            text = cache.read_text(encoding="utf-8")
        elif args.no_correct or not raw.strip():
            text = raw
        else:
            log(f"    morceau {i + 1}/{len(raws)}")
            text = correct_chunk(client, args.llm_model, raw, tail, slides, current, glossary)
            if word_count(text) < 0.75 * word_count(raw):
                log("    sortie trop courte, nouvel essai plus strict")
                text = correct_chunk(client, args.llm_model, raw, tail, slides, current, glossary, True)
                if word_count(text) < 0.7 * word_count(raw):
                    log("    toujours trop court : on garde la transcription brute pour ce morceau")
                    text = raw
            cache.write_text(text, encoding="utf-8")
        found = MARKER_RE.findall(text)
        if found:
            current = int(found[-1])
        tail = MARKER_RE.sub("", text)[-500:]
        corrected.append(text)

    # Assemblage
    body = []
    for (path, start), text in zip(chunks, corrected):
        body.append(f"`[{fmt_time(start)}]`\n\n{text}")
    full = render_markers("\n\n".join(body), slides)

    header = f"# Transcription : {args.video.stem}\n\n"
    if slides:
        header += "_Les séparateurs « Slide n » indiquent le passage d'une slide à l'autre (déduit du contenu), " \
                  "les horodatages `[hh:mm:ss]` marquent le début de chaque morceau audio._\n\n"
    out_path.write_text(header + full + "\n", encoding="utf-8")
    log(f"Terminé : {out_path}  ({word_count(full)} mots)")

    log(f"(cache dans {work}, supprime-le quand tu es satisfait)")


if __name__ == "__main__":
    main()
