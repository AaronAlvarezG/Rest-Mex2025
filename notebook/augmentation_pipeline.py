"""
Augmentation Pipeline — Rest-Mex 2025
======================================
Combina Back-Translation (Helsinki-NLP opus-mt) y Paráfrasis (T5)
para alcanzar ratio ~4:1 entre clase dominante y minoritarias.

Targets sobre el split de TRAIN (post simple_clean_v2 + filtro español):
    Clase 0 (Muy negativo): 3,804  → 24,000  (~20,196 nuevos)
    Clase 1 (Negativo):     3,840  → 24,000  (~20,160 nuevos)
    Clase 2 (Neutral):     10,856  → 31,000  (~20,144 nuevos)
    Clases 3 y 4: sin augmentation

Requisitos:
    pip install transformers sentencepiece langdetect scikit-learn torch

Modelos (se descargan automáticamente):
    Helsinki-NLP/opus-mt-es-en  |  Helsinki-NLP/opus-mt-en-es  (~300 MB c/u)
    Helsinki-NLP/opus-mt-es-fr  |  Helsinki-NLP/opus-mt-fr-es  (~300 MB c/u)
    mrm8488/t5-base-finetuned-spanish-summarization              (~900 MB)
"""

import pandas as pd
import numpy as np
import torch
import logging
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    MarianMTModel, MarianTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
)
from langdetect import detect, LangDetectException
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURACIÓN 
# ===========================================================================

# --- Rutas de entrada / salida ---------------------------------------------
TRAIN_PKL      = r'C:\Users\AaronMCC\Documents\Rest-Mex2025\dataset\split_train_15_03_2026_03_28_20.pkl'  # CSV del split de TRAIN (post-limpieza)
OUTPUT_CSV     = r'C:\Users\AaronMCC\Documents\Rest-Mex2025\resultados\train_augmented.csv'     # CSV aumentado resultante
FASTTEXT_PATH  = r'C:\Users\AaronMCC\Documents\Rest-Mex2025\dataset\cc.es.300.bin'           # Modelo FastText (opcional; deja "" para omitir)

# --- Columnas del DataFrame ------------------------------------------------
TEXT_COL  = "text_combined"   # Columna de texto (concatenación Title + Review limpia)
LABEL_COL = "label"           # Columna de etiqueta en escala 0-4

# --- Modelos de traducción (Helsinki-NLP MarianMT) -------------------------
MODEL_ES_EN = "Helsinki-NLP/opus-mt-es-en"
MODEL_EN_ES = "Helsinki-NLP/opus-mt-en-es"
MODEL_ES_FR = "Helsinki-NLP/opus-mt-es-fr"
MODEL_FR_ES = "Helsinki-NLP/opus-mt-fr-es"

# --- Modelo de paráfrasis T5 -----------------------------------------------
MODEL_T5 = "mrm8488/t5-base-finetuned-spanish-summarization"

# --- Targets de augmentation por clase (sobre el split de TRAIN) -----------
# Clases 3 y 4 no se aumentan (ya dominantes).
TARGET_CONFIG = {
    0: {"name": "Muy negativo", "original": 3_804,  "target": 24_000},
    1: {"name": "Negativo",     "original": 3_840,  "target": 24_000},
    2: {"name": "Neutral",      "original": 10_856, "target": 31_000},
}

# --- Distribución de métodos dentro de cada clase --------------------------
# Suma debe ser 1.0
METHOD_RATIOS = {
    "bt_en": 0.40,   # ES→EN→ES: variación sintáctica
    "bt_fr": 0.35,   # ES→FR→ES: preserva mejor el slang mexicano
    "t5":    0.25,   # Paráfrasis T5: diversidad semántica
}

# --- Hardware (RTX 3090 — 24 GB VRAM) --------------------------------------
# Todos los modelos caben simultáneamente en GPU (~6 GB total).
# Si ves CUDA OOM baja BATCH_SIZE_MT a 32 y BATCH_SIZE_T5 a 16.
BATCH_SIZE_MT = 48    # MarianMT (traducción)
BATCH_SIZE_T5 = 32    # T5 (paráfrasis, modelo más grande)

# --- Generación ------------------------------------------------------------
MAX_LENGTH       = 256    # Máximo de tokens por texto (entrada y salida)
NUM_BEAMS_MT     = 5      # Beam search para MarianMT
NUM_BEAMS_T5     = 5      # Beam search para T5
T5_TEMPERATURE   = 1.5    # Temperatura base T5 (>1.0 = más diversidad)
T5_TEMPERATURE_DELTA = 0.1  # Incremento de temperatura por pasada adicional
T5_TOP_P         = 0.95   # Nucleus sampling para T5

# --- Filtro de calidad -----------------------------------------------------
SIM_THRESHOLD_BT = 0.80   # Similitud coseno mínima para back-translation
SIM_THRESHOLD_T5 = 0.78   # Ligeramente más permisivo para paráfrasis T5

# --- Reproducibilidad ------------------------------------------------------
SEED = 42

# ===========================================================================
# FIN DE CONFIGURACIÓN — no es necesario modificar nada más abajo
# ===========================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
log.info(f"Dispositivo: {DEVICE}")


# ---------------------------------------------------------------------------
# 1. Traductores MarianMT (singleton por modelo)
# ---------------------------------------------------------------------------
class Translator:
    _cache: dict = {}

    @classmethod
    def get(cls, model_name: str) -> "Translator":
        if model_name not in cls._cache:
            cls._cache[model_name] = cls(model_name)
        return cls._cache[model_name]

    def __init__(self, model_name: str):
        log.info(f"Cargando traductor: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model     = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def translate(self, texts: list[str], num_beams: int = NUM_BEAMS_MT) -> list[str]:
        results = []
        for i in range(0, len(texts), BATCH_SIZE_MT):
            batch = texts[i : i + BATCH_SIZE_MT]
            with torch.no_grad():
                tokens = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).to(DEVICE)
                out = self.model.generate(**tokens, num_beams=num_beams)
            results.extend(
                self.tokenizer.batch_decode(out, skip_special_tokens=True)
            )
        return results


def back_translate_en(texts: list[str]) -> list[str]:
    """ES → EN → ES (variación sintáctica)."""
    en = Translator.get(MODEL_ES_EN).translate(texts)
    return Translator.get(MODEL_EN_ES).translate(en)


def back_translate_fr(texts: list[str]) -> list[str]:
    """ES → FR → ES (preserva mejor el slang mexicano)."""
    fr = Translator.get(MODEL_ES_FR).translate(texts)
    return Translator.get(MODEL_FR_ES).translate(fr)


# ---------------------------------------------------------------------------
# 2. Paráfrasis con T5
# ---------------------------------------------------------------------------
class T5Paraphraser:
    _instance = None

    @classmethod
    def get(cls) -> "T5Paraphraser":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        log.info(f"Cargando T5: {MODEL_T5}")
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_T5)
        self.model     = T5ForConditionalGeneration.from_pretrained(
            MODEL_T5
        ).to(DEVICE)
        self.model.eval()
        log.info(f"T5 cargado en {DEVICE} — BATCH_SIZE_T5={BATCH_SIZE_T5}")

    def paraphrase(
        self,
        texts: list[str],
        num_beams: int = NUM_BEAMS_T5,
        temperature: float = T5_TEMPERATURE,
        top_p: float = T5_TOP_P,
    ) -> list[str]:
        """
        Genera paráfrasis con muestreo diverso.
        temperature > 1.0 aumenta la diversidad del output.
        """
        results = []
        for i in range(0, len(texts), BATCH_SIZE_T5):
            batch = [f"paraphrase: {t}" for t in texts[i : i + BATCH_SIZE_T5]]
            with torch.no_grad():
                tokens = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).to(DEVICE)
                out = self.model.generate(
                    **tokens,
                    max_new_tokens=MAX_LENGTH,
                    num_beams=num_beams,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    early_stopping=True,
                )
            results.extend(
                self.tokenizer.batch_decode(out, skip_special_tokens=True)
            )
        return results


# ---------------------------------------------------------------------------
# 3. Filtro de calidad
# ---------------------------------------------------------------------------
def is_spanish(text: str) -> bool:
    try:
        return detect(text) == "es"
    except LangDetectException:
        return False


def compute_similarity_tfidf(
    originals: list[str],
    generated: list[str],
) -> np.ndarray:
    """TF-IDF char n-gram cosine similarity."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix     = vectorizer.fit_transform(originals + generated)
    orig_mat   = matrix[: len(originals)]
    gen_mat    = matrix[len(originals) :]
    return np.array([
        cosine_similarity(orig_mat[i], gen_mat[i])[0, 0]
        for i in range(len(originals))
    ])


def compute_similarity_fasttext(
    originals: list[str],
    generated: list[str],
    ft_model,
) -> np.ndarray:
    """Similitud coseno con embeddings FastText (promedio de tokens)."""
    def embed(text: str) -> np.ndarray:
        tokens = text.lower().split()
        vecs   = [ft_model.wv[t] for t in tokens if t in ft_model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(300)

    orig_vecs = np.array([embed(t) for t in originals])
    gen_vecs  = np.array([embed(t) for t in generated])
    return np.array([
        cosine_similarity(o.reshape(1, -1), g.reshape(1, -1))[0, 0]
        for o, g in zip(orig_vecs, gen_vecs)
    ])


def filter_generated(
    originals: list[str],
    generated: list[str],
    ft_model=None,
    threshold: float = SIM_THRESHOLD_BT,
) -> tuple[list[str], dict]:
    """
    Aplica tres filtros: idioma, similitud semántica, no-identidad.
    Retorna los textos generados que pasaron y estadísticas.
    """
    lang_mask = [is_spanish(g) for g in generated]
    sims      = (
        compute_similarity_fasttext(originals, generated, ft_model)
        if ft_model else
        compute_similarity_tfidf(originals, generated)
    )
    sim_mask  = sims >= threshold
    diff_mask = [
        g.strip().lower() != o.strip().lower()
        for o, g in zip(originals, generated)
    ]
    combined  = [l and s and d for l, s, d in zip(lang_mask, sim_mask, diff_mask)]

    valid = [g for g, m in zip(generated, combined) if m]
    stats = {
        "total":      len(originals),
        "pass_lang":  sum(lang_mask),
        "pass_sim":   sum(sim_mask),
        "pass_diff":  sum(diff_mask),
        "pass_all":   sum(combined),
        "retention":  sum(combined) / len(originals) if originals else 0,
    }
    return valid, stats


# ---------------------------------------------------------------------------
# 4. Generación por método con múltiples pasadas
# ---------------------------------------------------------------------------
def generate_with_method(
    texts: list[str],
    needed: int,
    method: str,
    ft_model=None,
    threshold: float = SIM_THRESHOLD_BT,
) -> tuple[list[str], list[str]]:
    """
    Genera `needed` ejemplos usando `method` ('bt_en', 'bt_fr', 't5').
    Hace múltiples pasadas sobre el corpus base con shuffling para diversidad.
    Retorna (valid_texts, method_labels).
    """
    valid   = []
    methods = []
    passes  = 0

    # Estimar cuántas pasadas necesitamos (asumiendo ~60% de retención)
    estimated_retention = 0.60
    max_passes          = int(np.ceil(needed / (len(texts) * estimated_retention))) + 3

    pbar = tqdm(total=needed, desc=f"  {method}", leave=False)

    while len(valid) < needed and passes < max_passes:
        # Shuffle diferente en cada pasada para diversidad
        rng    = np.random.RandomState(SEED + passes)
        batch  = rng.choice(texts, size=min(len(texts), needed - len(valid) + 50),
                            replace=False).tolist()

        if method == "bt_en":
            generated = back_translate_en(batch)
        elif method == "bt_fr":
            generated = back_translate_fr(batch)
        elif method == "t5":
            # Temperatura creciente en cada pasada para más diversidad
            temp      = T5_TEMPERATURE + (passes * T5_TEMPERATURE_DELTA)
            generated = T5Paraphraser.get().paraphrase(batch, temperature=temp)
        else:
            raise ValueError(f"Método desconocido: {method}")

        new_valid, stats = filter_generated(
            batch, generated, ft_model=ft_model, threshold=threshold
        )
        valid.extend(new_valid)
        methods.extend([method] * len(new_valid))
        pbar.update(len(new_valid))
        passes += 1

        log.debug(
            f"    Pasada {passes}: retención {stats['retention']:.1%} "
            f"({stats['pass_all']}/{stats['total']}) | "
            f"acumulado: {len(valid)}/{needed}"
        )

    pbar.close()
    return valid[:needed], methods[:needed]


# ---------------------------------------------------------------------------
# 5. Augmentation por clase
# ---------------------------------------------------------------------------
def augment_class(
    df_class: pd.DataFrame,
    target_n: int,
    ft_model=None,
) -> pd.DataFrame:
    """
    Aumenta una clase hasta `target_n` ejemplos combinando BT + T5.

    Estrategia de distribución:
        - 40% bt_en  (variación sintáctica)
        - 35% bt_fr  (preservación slang mexicano)
        - 25% t5     (paráfrasis semántica diversa)
    """
    texts     = df_class[TEXT_COL].tolist()
    original  = len(texts)
    needed    = target_n - original

    if needed <= 0:
        log.info(f"  Clase ya tiene {original} ejemplos, no requiere augmentation.")
        return pd.DataFrame(columns=[TEXT_COL, "augmentation_method"])

    log.info(f"  Generando {needed:,} nuevos ejemplos "
             f"(original: {original:,} → objetivo: {target_n:,})")

    # Cuotas por método
    n_en = int(needed * METHOD_RATIOS["bt_en"])
    n_fr = int(needed * METHOD_RATIOS["bt_fr"])
    n_t5 = needed - n_en - n_fr   # el resto a T5

    log.info(f"  Cuotas → bt_en: {n_en:,} | bt_fr: {n_fr:,} | t5: {n_t5:,}")

    all_texts   = []
    all_methods = []

    for method, quota, thresh in [
        ("bt_en", n_en, SIM_THRESHOLD_BT),
        ("bt_fr", n_fr, SIM_THRESHOLD_BT),
        ("t5",    n_t5, SIM_THRESHOLD_T5),
    ]:
        if quota <= 0:
            continue
        log.info(f"  [{method}] generando {quota:,} ...")
        valid, method_labels = generate_with_method(
            texts, quota, method=method, ft_model=ft_model, threshold=thresh
        )
        all_texts.extend(valid)
        all_methods.extend(method_labels)

        if len(valid) < quota:
            log.warning(
                f"  [{method}] solo generó {len(valid):,}/{quota:,}. "
                "Considera bajar SIM_THRESHOLD si el dataset final queda corto."
            )

    result = pd.DataFrame({
        TEXT_COL:               all_texts,
        "augmentation_method":  all_methods,
    })
    return result


# ---------------------------------------------------------------------------
# 6. Dry run — validación manual antes del pipeline completo
# ---------------------------------------------------------------------------
def dry_run(df_train: pd.DataFrame, n_samples: int = 5) -> None:
    """
    Genera n_samples ejemplos por clase y por método para revisión manual.
    Úsalo SIEMPRE antes del pipeline completo.
    """
    log.info("=" * 60)
    log.info("DRY RUN — revisión manual de calidad")
    log.info("=" * 60)

    for label_id, cfg in TARGET_CONFIG.items():
        subset = (
            df_train[df_train[LABEL_COL] == label_id]
            .sample(min(n_samples, len(df_train[df_train[LABEL_COL] == label_id])),
                    random_state=SEED)
        )
        texts = subset[TEXT_COL].tolist()

        log.info(f"\n--- Clase {label_id}: {cfg['name']} ---")

        for method, fn in [
            ("bt_en", back_translate_en),
            ("bt_fr", back_translate_fr),
            ("t5",    lambda t: T5Paraphraser.get().paraphrase(t)),
        ]:
            generated = fn(texts)
            print(f"\n  [{method}]")
            for orig, gen in zip(texts, generated):
                print(f"  ORIG: {orig[:100]}")
                print(f"  GEN:  {gen[:100]}")
                print()

    log.info("Revisa los textos anteriores antes de lanzar run_augmentation().")


# ---------------------------------------------------------------------------
# 7. Pipeline completo
# ---------------------------------------------------------------------------
def preload_all_models() -> None:
    """
    Precarga todos los modelos en GPU simultáneamente.
    Con RTX 3090 (24 GB) caben todos sin problema (~6 GB total):
        MarianMT ×4 ≈ ~1.2 GB c/u × 4 = ~4.8 GB
        T5 base   ×1 ≈ ~1.0 GB
    Hacer esto al inicio evita latencia durante la generación.
    """
    log.info("Precargando todos los modelos en GPU ...")
    Translator.get(MODEL_ES_EN)
    Translator.get(MODEL_EN_ES)
    Translator.get(MODEL_ES_FR)
    Translator.get(MODEL_FR_ES)
    T5Paraphraser.get()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        log.info(f"VRAM usada: {allocated:.1f} GB allocada / {reserved:.1f} GB reservada")
    log.info("Todos los modelos listos.")


def run_augmentation(
    df_train: pd.DataFrame,
    output_path: str = OUTPUT_CSV,
    ft_model=None,
    skip_dry_run: bool = False,
) -> pd.DataFrame:
    """
    Orquesta el pipeline completo de augmentation.

    Args:
        df_train:      DataFrame de TRAIN post-limpieza con columnas
                       'text_combined' y 'label' (0-4).
        output_path:   Ruta del CSV aumentado resultante.
        ft_model:      Modelo FastText de gensim (opcional).
        skip_dry_run:  Si False (default), ejecuta dry_run y pide confirmación.

    Returns:
        DataFrame combinado listo para entrenamiento.
    """
    assert TEXT_COL  in df_train.columns, f"Columna '{TEXT_COL}' no encontrada."
    assert LABEL_COL in df_train.columns, f"Columna '{LABEL_COL}' no encontrada."

    # Precargar todos los modelos en GPU antes de empezar
    preload_all_models()

    if not skip_dry_run:
        dry_run(df_train)
        resp = input("\n¿Continuar con el pipeline completo? [s/N]: ").strip().lower()
        if resp != "s":
            log.info("Cancelado.")
            return df_train

    all_augmented = []

    for label_id, cfg in TARGET_CONFIG.items():
        log.info(f"\n{'=' * 60}")
        log.info(f"Clase {label_id}: {cfg['name']}")
        log.info(f"{'=' * 60}")

        df_class = df_train[df_train[LABEL_COL] == label_id].copy()
        aug_df   = augment_class(df_class, target_n=cfg["target"], ft_model=ft_model)

        if len(aug_df) > 0:
            aug_df[LABEL_COL]       = label_id
            aug_df["is_augmented"]  = True
            all_augmented.append(
                aug_df[[TEXT_COL, LABEL_COL, "is_augmented", "augmentation_method"]]
            )
            log.info(f"  Generados: {len(aug_df):,} nuevos ejemplos")

    # Combinar con datos originales
    df_orig                     = df_train.copy()
    df_orig["is_augmented"]     = False
    df_orig["augmentation_method"] = "original"

    if all_augmented:
        df_aug      = pd.concat(all_augmented, ignore_index=True)
        df_combined = pd.concat([df_orig, df_aug], ignore_index=True)
    else:
        df_combined = df_orig

    df_combined = df_combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Reporte final
    log.info("\n" + "=" * 60)
    log.info("DISTRIBUCIÓN FINAL DEL DATASET DE TRAIN AUMENTADO")
    log.info("=" * 60)
    total_aug = 0
    for label_id, cfg in TARGET_CONFIG.items():
        orig  = cfg["original"]
        final = (df_combined[LABEL_COL] == label_id).sum()
        added = final - orig
        total_aug += added
        log.info(
            f"  {cfg['name']:15s} (clase {label_id}): "
            f"{orig:>7,} → {final:>7,}  (+{added:,})"
        )
    for label_id in [3, 4]:
        n = (df_combined[LABEL_COL] == label_id).sum()
        log.info(f"  Clase {label_id} (sin aug):      {n:>7,}")

    log.info(f"\n  Total train original:  {len(df_train):>7,}")
    log.info(f"  Total train aumentado: {len(df_combined):>7,}  (+{total_aug:,})")
    log.info(
        f"  Ratio final Muy positivo:Muy negativo = "
        f"{(df_combined[LABEL_COL] == 4).sum():,}:"
        f"{(df_combined[LABEL_COL] == 0).sum():,}"
    )

    # Desglose por método
    log.info("\n  Desglose por método de augmentation:")
    for method, count in df_combined["augmentation_method"].value_counts().items():
        log.info(f"    {method:12s}: {count:>8,}")

    df_combined.to_csv(output_path, index=False)
    log.info(f"\nGuardado en: {output_path}")

    return df_combined


# ---------------------------------------------------------------------------
# 8. Carga para integración con DataLoader existente
# ---------------------------------------------------------------------------
def load_augmented_for_training(
    csv_path: str,
) -> tuple[list[str], list[int]]:
    """
    Carga el CSV aumentado y devuelve listas para tu Dataset.
    Solo usa columnas TEXT_COL y LABEL_COL definidas en la configuración.

    Ejemplo:
        texts, labels = load_augmented_for_training(OUTPUT_CSV)
        train_dataset = RestMexDataset(texts, labels, ...)
    """
    df     = pd.read_csv(csv_path)
    texts  = df[TEXT_COL].tolist()
    labels = df[LABEL_COL].tolist()
    log.info(
        f"Cargados {len(texts):,} ejemplos — "
        f"distribución: {pd.Series(labels).value_counts().sort_index().to_dict()}"
    )
    return texts, labels


# ---------------------------------------------------------------------------
# 9. Punto de entrada
# ---------------------------------------------------------------------------
# Compatible con ejecución desde Jupyter (ignora argumentos internos del
# kernel como -f kernel.json) y desde CLI estándar.
if __name__ == "__main__":
    import argparse
    import sys

    # Detectar si se está ejecutando desde Jupyter
    _in_jupyter = any("ipykernel" in arg or arg.endswith(".json")
                      for arg in sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Augmentation pipeline BT + T5 para Rest-Mex 2025"
    )
    parser.add_argument("--train_pkl",     default=TRAIN_PKL,
                        help="PKL del split de TRAIN (post-limpieza)")
    parser.add_argument("--output_csv",    default=OUTPUT_CSV)
    parser.add_argument("--dry_run",       action="store_true",
                        help="Solo validación manual, no genera el dataset completo")
    parser.add_argument("--skip_confirm",  action="store_true",
                        help="No pide confirmación (útil para scripts)")
    parser.add_argument("--fasttext_path", default=FASTTEXT_PATH,
                        help="Ruta al modelo FastText .bin (opcional)")

    # parse_known_args ignora argumentos desconocidos (ej. -f kernel.json de Jupyter)
    args, unknown = parser.parse_known_args()
    if unknown:
        log.info(f"Argumentos ignorados (entorno Jupyter): {unknown}")

    df_train = pd.read_pickle(args.train_pkl)
    log.info(f"Train cargado: {len(df_train):,} filas")

    # Intentar cargar FastText para similitud semántica más precisa
    ft_model = None
    if Path(args.fasttext_path).exists():
        try:
            from gensim.models.fasttext import load_facebook_model
            log.info(f"Cargando FastText desde {args.fasttext_path} ...")
            ft_model = load_facebook_model(args.fasttext_path)
            log.info("FastText cargado — usando similitud coseno sobre embeddings.")
        except Exception as e:
            log.warning(f"No se pudo cargar FastText ({e}). Usando TF-IDF de respaldo.")
    else:
        log.info("FastText no encontrado — usando TF-IDF char n-gram como respaldo.")

    if args.dry_run:
        dry_run(df_train)
    else:
        run_augmentation(
            df_train,
            output_path=args.output_csv,
            ft_model=ft_model,
            skip_dry_run=args.skip_confirm,
        )
