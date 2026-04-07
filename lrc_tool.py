# LRC Text Differentiator — Proof of Concept (enhanced)
# -----------------------------------------------------
# Differentiates human vs. AI-generated text using Long-Range Correlations (LRC).
# Encodings:
#   1) Multi-target word presence family: OR / inter-arrival gaps / windowed density
#   2) Sentence-length series (words per sentence)
#   3) Function-word indicator series
#   4) Punctuation-cadence (to determine long-range structural rhythm)
#   5) Semantic drift (embedding-space sentence-to-sentence distance)
#
# Enhancements vs original:
# - Multi-keyword targeting: --targets "the,of,and" and --presence-mode {or,gap,density}
# - Constant/short-series guards for stable DFA/Hurst
# - Optional DFA-only mode for comparability
# - Batch-safe semantic drift with sentence cap (--max-sents)
# - Improved sentence splitter for quotes/brackets/ellipses
# - Optional surrogate tests (--surrogates) with ΔH reporting
# - Weighted aggregation and series length reporting
#
# Usage examples:
#   python the_code.py
#   python the_code.py --text1 file.txt --targets "the,of,and" --presence-mode density --window 128
#   python the_code.py --dfa-only --surrogates --weights "1,1,1,0.8,1.2"
#
# Dependencies:
#   python -m pip install -U numpy scipy scikit-learn nolds sentence-transformers "torch>=2.3,<3"

import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
# Context-fit (MLM) imports
from transformers import pipeline
import numpy as np
import nolds

# ---------------------- Config ----------------------
MIN_RS_POINTS = 500  # min series length to use RS Hurst; else DFA fallback
TARGET_WORD_DEFAULT = "joyce,myth,mythology,Nietzsche,Vico,science,reason"

FUNCTION_WORDS = {
    "a","an","the","and","or","but","if","then","else","for","nor","so","yet",
    "of","in","on","at","by","to","from","with","without","about","as","into","than","because",
    "is","am","are","was","were","be","been","being",
    "do","does","did","doing","done",
    "have","has","had","having",
    "this","that","these","those","it","its","itself",
    "he","him","his","she","her","hers","they","them","their","theirs","we","us","our","ours",
    "you","your","yours","i","me","my","mine",
    "not","no","yes","also","very","too","just",
    "there","here","when","where","why","how","though","while","over","under","through","between","against"
}
# Punctuation set for cadence/burstiness (includes . , ; : ! ? quotes, parentheses, ellipsis …, en dash – , em dash — , hyphen -)
PUNCT_CHARS = set(list(".,;:!?()—\"'—–-…—"))
# ------------------- Preprocessing ------------------

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())

def split_sentences(text: str) -> List[str]:
    # Split on whitespace that follows a sentence end (. ! ? …) or on newlines
    text = text.strip()
    parts = re.split(r'(?<=[.!?…])\s+|\n+', text)
    return [s.strip() for s in parts if s and s.strip()]

# ------------------- Embeddings ---------------------

try:
    from sentence_transformers import SentenceTransformer
    _sem_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    _sem_model = None
    _sem_err = e

def series_semantic_drift(sentences: List[str], max_sents: int = 5000, batch_size: int = 64) -> np.ndarray:
    if len(sentences) < 2:
        return np.array([0.0])
    if _sem_model is None:
        raise RuntimeError(f"SentenceTransformer unavailable: {repr(_sem_err)}")
    sents = sentences[:max_sents]
    # Normalized embeddings -> cosine = dot
    embeddings = _sem_model.encode(
        sents, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=batch_size, show_progress_bar=False
    )
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    drift = 1.0 - sims
    return drift.astype(float)

# ---------------------- Encoders --------------------

# Masked-LM for context-fit (RoBERTa; robust tokenization)
try:
    _mask = pipeline("fill-mask", model="roberta-base", device=-1)
    MASK_TOKEN = _mask.tokenizer.mask_token  # usually "<mask>"
except Exception as e:
    _mask = None
    _mask_err = e

def series_word_presence_multi(words: List[str], targets: List[str]) -> np.ndarray:
    T = set(t.lower() for t in targets)
    return np.array([1.0 if w in T else 0.0 for w in words], dtype=float)

def series_word_interarrival_multi(words: List[str], targets: List[str]) -> np.ndarray:
    T = set(t.lower() for t in targets)
    idx = [i for i, w in enumerate(words) if w in T]
    if len(idx) < 2:
        return np.array([0.0])
    gaps = np.diff(idx).astype(float)
    return np.log1p(gaps)  # stabilize heavy tails

def series_word_density_multi(words: List[str], targets: List[str], window:int=128) -> np.ndarray:
    x = series_word_presence_multi(words, targets)
    if len(x) == 0:
        return x
    window = max(4, min(window, len(x)))
    k = np.ones(window, dtype=float) / window
    dens = np.convolve(x, k, mode="same")
    return dens.astype(float)

def series_sentence_lengths(sentences: List[str]) -> np.ndarray:
    lengths = [len(tokenize_words(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]
    return np.array(lengths, dtype=float)

def series_function_word_indicator(words: List[str]) -> np.ndarray:
    return np.array([1.0 if w in FUNCTION_WORDS else 0.0 for w in words], dtype=float)

def series_punct_counts_per_sentence(text: str, punct_chars: Optional[set] = None) -> np.ndarray:
    """Count defined punctuation marks per sentence; returns a real-valued series."""
    if punct_chars is None:
        punct_chars = PUNCT_CHARS
    sents = split_sentences(text)
    counts = []
    for s in sents:
        c = sum(1 for ch in s if ch in punct_chars)
        counts.append(c)
    return np.array(counts, dtype=float)

def fano_factor(x: np.ndarray) -> float:
    """Variance-to-mean ratio with guards (0 if mean≈0)."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    m = float(np.mean(x))
    v = float(np.var(x))
    if m <= 1e-12:
        return 0.0
    return v / m

# ------------------ Surrogates ----------------------

def block_shuffle(x: np.ndarray, block: int = 50, seed: int = 0) -> np.ndarray:
    x = np.asarray(x).ravel()
    n = len(x)
    if n <= block:
        return x.copy()
    blocks = [x[i:i+block] for i in range(0, n, block)]
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    return np.concatenate(blocks)

def phase_randomize(y: np.ndarray, seed: int = 0) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    Y = np.fft.rfft(y)
    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(0, 2*np.pi, size=Y.shape))
    phases[0] = 1.0
    if Y.shape[0] % 2 == 0:
        phases[-1] = 1.0  # keep Nyquist real
    y_surr = np.fft.irfft(Y * phases, n=len(y)).real
    return y_surr

def context_fit_scores(
    text: str,
    terms: List[str],
    topk: int = 10,
    max_evals: int = 200
) -> List[float]:
    """
    For each occurrence of any term in `terms`, mask that occurrence in its sentence
    and ask the MLM how well the original word fits the context.
    Returns scores in [0,1], where 1 = top-1 prediction, 0 = not in top-k.
    """
    if _mask is None:
        raise RuntimeError(f"Context-fit unavailable: {repr(_mask_err)}")

    sentences = split_sentences(text)
    scores: List[float] = []
    T = {t.lower() for t in terms}

    for s in sentences:
        # find exact word hits (case-insensitive), one at a time
        words = re.findall(r"\b[\w']+\b", s)
        lower = [w.lower() for w in words]
        for i, w in enumerate(lower):
            if w in T:
                # mask just this occurrence
                masked = re.sub(rf"\b{re.escape(words[i])}\b", MASK_TOKEN, s, count=1)
                try:
                    preds = _mask(masked, top_k=topk)
                except Exception:
                    continue
                # rank of the original word among top-k predictions
                rank = topk + 1
                for idx, p in enumerate(preds, start=1):
                    cand = p["token_str"].strip().lower()
                    if cand == w:
                        rank = idx
                        break
                score = 1.0 - (rank - 1) / topk if rank <= topk else 0.0
                scores.append(score)
                if len(scores) >= max_evals:
                    return scores
    return scores

# ------------------ Hurst / DFA Safe ----------------

def hurst_or_dfa(x: np.ndarray, min_rs_points: int = MIN_RS_POINTS, dfa_only: bool = False) -> Tuple[float, str]:
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n == 0:
        return 0.5, "NA(empty)"
    if np.allclose(x, x[0]):
        return 0.5, "NA(constant)"
    std = np.std(x)
    x = (x - np.mean(x)) / (std + 1e-12)
    if dfa_only:
        try:
            return float(nolds.dfa(x)), "DFA"
        except Exception:
            return 0.5, "DFA(error)"
    if n >= min_rs_points:
        try:
            return float(nolds.hurst_rs(x)), "RS"
        except Exception:
            try:
                return float(nolds.dfa(x)), "DFA(fallback)"
            except Exception:
                return 0.5, "DFA(error)"
    else:
        try:
            return float(nolds.dfa(x)), "DFA"
        except Exception:
            return 0.5, "DFA(error)"

# --------------------- Aggregation ------------------

@dataclass
class EncodingStat:
    value: float
    method: str
    n: int
    surrogate: Optional[float] = None
    delta: Optional[float] = None

@dataclass
class LRCResult:
    encoding_values: Dict[str, EncodingStat]  # encoding -> stats
    aggregate_score: float
    verdict: str
    # extras (not in weighted aggregate)
    punc_fano: Optional[float] = None
    punc_n: Optional[int] = None

def aggregate_verdict(exponents: List[float], weights: Optional[List[float]] = None) -> Tuple[float, str]:
    exps = np.array(exponents, dtype=float)
    if weights is None:
        agg = float(np.mean(exps))
    else:
        w = np.array(weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        agg = float(np.sum(exps * w))
    if agg >= 0.56:
        verdict = "More human-like global memory (persistent)."
    elif agg <= 0.50:
        verdict = "More AI-like or weak global memory (anti-persistent)."
    else:
        verdict = "Ambiguous / mixed global memory."
    return agg, verdict

# --------------------- Main API ---------------------

def compute_encoding(x: np.ndarray, min_rs_points: int, dfa_only: bool, use_surrogates: bool,
                     series_kind: str) -> EncodingStat:
    """series_kind: 'indicator' or 'real' (affects surrogate choice)"""
    n = int(len(x))
    val, method = hurst_or_dfa(x, min_rs_points=min_rs_points, dfa_only=dfa_only)
    sur = delt = None
    if use_surrogates and n >= 16:
        if series_kind == 'real':
            xs = phase_randomize(x)
        else:
            block = max(10, min(50, n // 10))
            xs = block_shuffle(x, block=block)
        s_val, _ = hurst_or_dfa(xs, min_rs_points=min_rs_points, dfa_only=dfa_only)
        sur = float(s_val)
        delt = float(val - sur)
    return EncodingStat(value=float(val), method=method, n=n, surrogate=sur, delta=delt)

def analyze_text(
    text: str,
    target_word: str = TARGET_WORD_DEFAULT,
    min_rs_points: int = MIN_RS_POINTS,
    targets: Optional[List[str]] = None,
    presence_mode: str = "or",
    window: int = 128,
    dfa_only: bool = False,
    use_surrogates: bool = False,
    max_sents: int = 5000
) -> LRCResult:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not targets:
        targets = [target_word.lower()]

    encodings: Dict[str, EncodingStat] = {}

    # 1) Multi-target presence family
    if presence_mode == "or":
        x = series_word_presence_multi(words, targets)
        presence_name = "word_presence_any"
        kind = 'indicator'
    elif presence_mode == "gap":
        x = series_word_interarrival_multi(words, targets)
        presence_name = "word_interarrival_any"
        kind = 'real'
    else:  # density
        x = series_word_density_multi(words, targets, window=window)
        presence_name = f"word_density_any(w={window})"
        kind = 'real'
    encodings[presence_name] = compute_encoding(
        x, min_rs_points, dfa_only, use_surrogates, kind
    )

    # 2) Sentence lengths
    sl = series_sentence_lengths(sentences)
    encodings["sentence_lengths"] = compute_encoding(
        sl, min_rs_points, dfa_only, use_surrogates, 'real'
    )

    # 3) Function-word indicator
    fw = series_function_word_indicator(words)
    encodings["function_words"] = compute_encoding(
        fw, min_rs_points, dfa_only, use_surrogates, 'indicator'
    )

    # 4) Punctuation cadence (per-sentence counts -> DFA/RS) + Fano (burstiness)
    pc = series_punct_counts_per_sentence(text)
    encodings["punctuation_cadence"] = compute_encoding(
        pc, min_rs_points, dfa_only, use_surrogates, 'real'
    )
    punc_fano_val = fano_factor(pc)

    # 5) Semantic drift between adjacent sentences
    sd = series_semantic_drift(sentences, max_sents=max_sents)
    encodings["semantic_drift"] = compute_encoding(
        sd, min_rs_points, dfa_only, use_surrogates, 'real'
    )

    # Aggregation (respect presence encoding first)
    order = [presence_name, "sentence_lengths", "function_words", "punctuation_cadence", "semantic_drift"]
    exps = [encodings[name].value for name in order]
    agg, verdict = aggregate_verdict(exps)

    return LRCResult(
        encoding_values=encodings,
        aggregate_score=agg,
        verdict=verdict,
        punc_fano=punc_fano_val,
        punc_n=int(len(pc))
    )

def compare_texts(
    text1: str, text2: str,
    target_word: str = TARGET_WORD_DEFAULT,
    min_rs_points: int = MIN_RS_POINTS,
    targets: Optional[List[str]] = None,
    presence_mode: str = "or",
    window: int = 128,
    dfa_only: bool = False,
    use_surrogates: bool = False,
    weights: Optional[List[float]] = None,
    max_sents: int = 5000,
    # context-fit args
    context_terms: Optional[str] = None,
    context_topk: int = 10,
    context_maxevals: int = 200,
) -> None:
    # analyze both texts
    r1 = analyze_text(text1, target_word, min_rs_points, targets, presence_mode, window,
                      dfa_only, use_surrogates, max_sents)
    r2 = analyze_text(text2, target_word, min_rs_points, targets, presence_mode, window,
                      dfa_only, use_surrogates, max_sents)

    # ------- pretty-printer (NO recursion here) -------
    def print_block(res: LRCResult, label: str, weights_opt: Optional[List[float]]):
        print(f"\n=== {label} ===")
        for name, stats in res.encoding_values.items():
            line = f"{name:24s}: {stats.value:.3f}  [{stats.method}]  (n={stats.n})"
            if getattr(stats, "delta", None) is not None and getattr(stats, "surrogate", None) is not None:
                line += f"  ΔH={stats.delta:+.3f} (sur={stats.surrogate:.3f})"
            print(line)

 # Punctuation burstiness (Fano) — extra, not in weighted aggregate
        if getattr(res, "punc_fano", None) is not None:
            n_punc = res.encoding_values.get("punctuation_cadence", EncodingStat(0,"",0)).n
            print(f"{'punctuation_fano':24s}: {res.punc_fano:.3f}  (n={n_punc})")

        # aggregate (weighted if provided)
        presence_keys = [k for k in res.encoding_values.keys() if k.startswith("word_")]
        agg_order = presence_keys + ["sentence_lengths", "function_words", "punctuation_cadence", "semantic_drift"]
        exps = [res.encoding_values[k].value for k in agg_order if k in res.encoding_values]
        if weights_opt is not None and len(weights_opt) == len(exps):
            w = np.array(weights_opt, dtype=float)
            w = w / (np.sum(w) + 1e-12)
            agg = float(np.sum(np.array(exps) * w))
        else:
            agg = float(np.mean(exps))
        print(f"Aggregate (mean/weighted): {agg:.3f}")
        print(f"Verdict: {res.verdict}")

    # ------- context-fit helper -------
    def _print_context_fit(label: str, txt: str):
        if not context_terms:
            return
        terms = [t.strip() for t in context_terms.split(",") if t.strip()]
        try:
            cfs = context_fit_scores(txt, terms, topk=context_topk, max_evals=context_maxevals)
            if cfs:
                print(f"Context Fit ({label}, mean over {len(cfs)} hits): {np.mean(cfs):.3f}  [1=strong, 0=weak]")
            else:
                print(f"Context Fit ({label}): no occurrences of provided terms found.")
        except Exception as e:
            print(f"Context Fit ({label}): unavailable ({e})")

    # ------- print results -------
    print_block(r1, "Text 1", weights)
    _print_context_fit("Text 1", text1)

    print_block(r2, "Text 2", weights)
    _print_context_fit("Text 2", text2)

    # comparison summary stays last
    print("\n--- Comparison Summary ---")
    if r1.aggregate_score > r2.aggregate_score + 0.02:
        print("Text 1 shows stronger long-range persistence (more human-like).")
    elif r2.aggregate_score > r1.aggregate_score + 0.02:
        print("Text 2 shows stronger long-range persistence (more human-like).")
    else:
        print("Both texts are similar in global memory; use local features or context-fit for tie-break.")

# --------------------- CLI / Tests ------------------

HUMAN_TEXT = """

"""

GPT_TEXT   = """

"""

def parse_targets(args) -> List[str]:
    if args.targets:
        return [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    return [args.target.lower()]

def main():
    parser = argparse.ArgumentParser(description="LRC Text Differentiator — Proof of Concept (enhanced)")
    parser.add_argument("--context-terms", type=str,
                        help='Comma-separated terms to assess context fit (e.g., "myth,history,he")')
    parser.add_argument("--context-topk", type=int, default=10, help="Top-K predictions to consider for context fit")
    parser.add_argument("--context-maxevals", type=int, default=200, help="Max occurrences to score for context fit")
    parser.add_argument("--text1", type=str, help="Path to first text file (optional)")
    parser.add_argument("--text2", type=str, help="Path to second text file (optional)")
    parser.add_argument("--target", type=str, default=TARGET_WORD_DEFAULT, help="Single target word (fallback)")
    parser.add_argument("--targets", type=str, help="Comma-separated list of target words; overrides --target")
    parser.add_argument("--presence-mode", choices=["or","gap","density"], default="or",
                        help="Encoding for multi-target presence: OR (0/1), inter-arrival gaps, or windowed density")
    parser.add_argument("--window", type=int, default=128, help="Window size for density mode (tokens)")
    parser.add_argument("--min-rs", type=int, default=MIN_RS_POINTS, help="Minimum series length to use RS Hurst")
    parser.add_argument("--dfa-only", action="store_true", help="Force DFA for all encodings")
    parser.add_argument("--surrogates", action="store_true", help="Compute surrogate ΔH for each encoding")
    parser.add_argument("--weights", type=str, help="Comma-separated weights (presence, sentlen, func, punct, drift)")
    parser.add_argument("--max-sents", type=int, default=5000, help="Cap on sentences for semantic drift")
    args = parser.parse_args()

    # Load texts
    def load(p, default):
        if p:
            key = p.strip().upper()
            if key == "HUMAN_TEXT":
                return HUMAN_TEXT
            if key == "GPT_TEXT":
                return GPT_TEXT
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        return default  # <- make sure this is aligned with 'def', not inside 'with'

    t1 = load(args.text1, HUMAN_TEXT)
    t2 = load(args.text2, GPT_TEXT)

    targets = parse_targets(args)  # assumes you already have this helper

    # Optional weights parsing (must match 5 encodings in order)
    weights = None
    if args.weights:
        try:
            w = [float(x) for x in args.weights.split(",")]
            if len(w) == 5:
                weights = w
        except Exception:
            pass

    if t1 and t2:
        # --- Compare mode ---
        compare_texts(
            t1, t2,
            target_word=args.target,
            min_rs_points=args.min_rs,
            targets=targets,
            presence_mode=args.presence_mode,
            window=args.window,
            dfa_only=args.dfa_only,
            use_surrogates=args.surrogates,
            weights=weights,
            max_sents=args.max_sents,
            context_terms=args.context_terms,
            context_topk=args.context_topk,
            context_maxevals=args.context_maxevals,
        )

    elif t1 and not t2:
        # --- Single Text Analysis ---
        res = analyze_text(
            t1,
            target_word=args.target,
            min_rs_points=args.min_rs,
            targets=targets,
            presence_mode=args.presence_mode,
            window=args.window,
            dfa_only=args.dfa_only,
            use_surrogates=args.surrogates,
            max_sents=args.max_sents
        )
        print("\n=== Single Text Analysis ===")
        for name, stats in res.encoding_values.items():
            line = f"{name:24s}: {stats.value:.3f}  [{stats.method}]  (n={stats.n})"
            if getattr(stats, "delta", None) is not None and getattr(stats, "surrogate", None) is not None:
                line += f"  ΔH={stats.delta:+.3f} (sur={stats.surrogate:.3f})"
            print(line)
        if getattr(res, "punc_fano", None) is not None:
            n_punc = res.encoding_values.get("punctuation_cadence", EncodingStat(0,"",0)).n
            print(f"{'punctuation_fano':24s}: {res.punc_fano:.3f}  (n={n_punc})")
        print(f"Aggregate (mean): {res.aggregate_score:.3f}")
        print(f"Verdict: {res.verdict}")

        # Optional Context Fit for single text only
        if args.context_terms:
            terms = [t.strip() for t in args.context_terms.split(",") if t.strip()]
            try:
                cfs = context_fit_scores(t1, terms, topk=args.context_topk, max_evals=args.context_maxevals)
                if cfs:
                    print(f"Context Fit (Single Text, mean over {len(cfs)} hits): {np.mean(cfs):.3f}  [1=strong, 0=weak]")
                else:
                    print("Context Fit: no occurrences of provided terms found.")
            except Exception as e:
                print(f"Context Fit: unavailable ({e})")

    else:
        # --- Default: built-in comparison ---
        compare_texts(
            HUMAN_TEXT, GPT_TEXT,
            target_word=args.target,
            min_rs_points=args.min_rs,
            targets=targets,
            presence_mode=args.presence_mode,
            window=args.window,
            dfa_only=args.dfa_only,
            use_surrogates=args.surrogates,
            weights=weights,
            max_sents=args.max_sents,
            context_terms=args.context_terms,
            context_topk=args.context_topk,
            context_maxevals=args.context_maxevals,
        )

if __name__ == "__main__":
    main()

