import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing  import Any, Dict, List, Optional, Tuple
import numpy as np
import joblib
import xgboost as xgb
from flask   import Flask, request, jsonify
from flask_cors import CORS

# LOGGING
logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt  = '%Y-%m-%d %H:%M:%S',
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler('predict_api.log', encoding='utf-8'),
    ],
)
log = logging.getLogger('predict-api')

# ────────────────────────Configuration et constantes────────────────────────
_BASE_DIR = Path(__file__).resolve().parent

CONFIG: Dict[str, Any] = {
    'USE_MONGODB'        : True,
    'MONGO_URI'          : 'mongodb://localhost:27017',
    'MONGO_DB'           : 'test',
    'MONGO_COL_CONTACTS' : 'contacts',
    'JSON_USER_CONTACT'  : str(_BASE_DIR / 'user_contact.json'),
    'MODEL_PKL'          : str(_BASE_DIR / 'outputs' / 'model_assets.joblib'),
    'MODEL_XGB'          : str(_BASE_DIR / 'outputs' / 'xgboost_model.json'),
    'HOST'               : '0.0.0.0',
    'PORT'               : 5001,
    'DEBUG'              : False,
    'DEFAULT_THR_AVAILABLE' : 55,
    'DEFAULT_THR_SUSPECTED' : 35,
    'DEFAULT_W_XGB_SHORT'   : 0.65,
    'DEFAULT_W_RSF_SHORT'   : 0.35,
    'DEFAULT_W_XGB_LONG'    : 0.30,
    'DEFAULT_W_RSF_LONG'    : 0.70,
    'DEFAULT_W_SHORT'       : 0.50,
    'DEFAULT_W_LONG'        : 0.50,
    'DEFAULT_AVAIL_CONFIDENT_THR': 0.85,
    'DEFAULT_AVAIL_REDUCTION'    : 0.30,
    'DEFAULT_NA_CONFIDENT_THR'   : 0.80,
    'DEFAULT_NA_CONFIDENT_W'     : 0.70,
    'DEFAULT_DIVERGENCE_THR'     : 0.40,
    'DEFAULT_DIVERGENCE_W_XGB'   : 0.60,
    'DEFAULT_DIVERGENCE_W_RSF'   : 0.40,
    'MSISDN_MIN_LEN'     : 8,
    'MSISDN_MAX_LEN'     : 15,
    'MAX_MSISDNS_PER_REQ': 5_000,
}

# ── Features XGBoost : identiques à XGB_FEATURE_COLS dans train_model.py ──
XGB_FEATURE_COLS: List[str] = [
    'n_envois_hist',
    'n_succes_hist',
    'n_echecs_hist',
    'echecs_consecutifs_max',
    'echecs_consecutifs_fin',
    'jours_depuis_dernier_envoi',
    'jours_depuis_succes',
    'score_recence_pondere',
    'freq_inter_envoi_jours',
    'duree_observation_jours',
    'n_succes_30j',
    'n_echecs_30j',
    'taux_succes_30j',
]

# ── Features RSF : identiques à RSF_FEATURE_COLS dans train_model.py ──
RSF_FEATURE_COLS: List[str] = [
    'n_envois_hist',
    'n_succes_hist',
    'echecs_consecutifs_fin',
    'jours_depuis_dernier_envoi',
    'jours_depuis_succes',
    'score_recence_pondere',
    'n_succes_30j',
    'taux_succes_30j',
]

# Plages de valeurs acceptables par feature (pour /api/predict_direct)
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    'n_envois_hist'              : (0, 100_000),
    'n_succes_hist'              : (0, 100_000),
    'n_echecs_hist'              : (0, 100_000),
    'echecs_consecutifs_max'     : (0, 10_000),
    'echecs_consecutifs_fin'     : (0, 10_000),
    'jours_depuis_dernier_envoi' : (0, 9_999),
    'jours_depuis_succes'        : (0, 9_999),
    'score_recence_pondere'      : (-10.0, 10.0),
    'freq_inter_envoi_jours'     : (0.0, 10_000),
    'duree_observation_jours'    : (0, 9_999),
    'n_succes_30j'               : (0, 10_000),
    'n_echecs_30j'               : (0, 10_000),
    'taux_succes_30j'            : (0.0, 1.0),
}

# Constantes DLR (alignées avec train_model.py)
DLR_SUCCESS = frozenset({1})
DLR_FAILURE = frozenset({2})
DLR_NA      = frozenset({0, 16, 34})
DLR_TRANSIT = frozenset({4, 8})
DLR_FINAL   = DLR_SUCCESS | DLR_FAILURE
DLR_NORM    = {1: 1, 2: 2, 4: 2, 8: 2}

# Regex MSISDN
_MSISDN_RE = re.compile(r'^\+?[0-9]{8,15}$')


# ═════════════════════ Singleton thread-safe : ModelRegistry ═════════════════
class _ModelRegistry:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._pkg: Optional[Dict] = None
        self._pkg_sha: Optional[str] = None
        self._loaded_at: Optional[datetime] = None

    @property
    def pkg(self) -> Dict:
        if self._pkg is None:
            self._load()
        return self._pkg  # type: ignore[return-value]

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    def _load(self) -> None:
        pkl_path = CONFIG['MODEL_PKL']
        with self._lock:
            if self._pkg is not None:
                return
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f"Modèle introuvable : {pkl_path}\n"
                    f"Lancez train_model.py d'abord."
                )
            log.info(f"Chargement du modèle → {pkl_path}")
            raw = open(pkl_path, 'rb').read()
            self._pkg_sha   = hashlib.sha256(raw).hexdigest()[:12]
            self._pkg       = joblib.load(pkl_path)

            xgb_path = CONFIG['MODEL_XGB']
            if os.path.exists(xgb_path):
                xgb_clf = xgb.XGBClassifier()
                xgb_clf.load_model(xgb_path)
                self._pkg['xgboost_model'] = xgb_clf
                log.info(f"Modèle XGBoost chargé → {xgb_path}")
            else:
                raise FileNotFoundError(
                    f"Modèle XGBoost introuvable : {xgb_path}\n"
                    f"Lancez train_model.py d'abord."
                )
            self._loaded_at = datetime.now()
            self._validate_pkg()
            log.info(
                f"Modèle chargé  "
                f"v={self._pkg.get('version','?')}  "
                f"sha={self._pkg_sha}  "
                f"entraîné={self._pkg.get('trained_at','?')}"
            )
            self._log_thresholds()

    def _validate_pkg(self) -> None:
        pkg = self._pkg
        missing = [k for k in ('xgboost_model', 'label_map') if k not in pkg]
        if missing:
            raise ValueError(f"Modèle .pkl incomplet — clés manquantes : {missing}")

        # Vérification des colonnes features sauvegardées
        saved_xgb_cols = pkg.get('xgb_feature_cols', [])
        if saved_xgb_cols and saved_xgb_cols != XGB_FEATURE_COLS:
            log.warning(
                f"ATTENTION : xgb_feature_cols PKL={saved_xgb_cols} "
                f"≠ XGB_FEATURE_COLS API={XGB_FEATURE_COLS}. "
                f"Utilisation des colonnes sauvegardées dans le PKL."
            )

        if 'score_thresholds' not in pkg:
            log.warning(
                "pkg['score_thresholds'] absent → seuils par défaut utilisés "
                f"(available={CONFIG['DEFAULT_THR_AVAILABLE']}, "
                f"suspected={CONFIG['DEFAULT_THR_SUSPECTED']})."
            )
        else:
            thr = pkg['score_thresholds']
            av = int(thr.get('available', CONFIG['DEFAULT_THR_AVAILABLE']))
            su = int(thr.get('suspected', CONFIG['DEFAULT_THR_SUSPECTED']))
            if av <= su:
                log.warning(
                    f"Incohérence seuils : available({av}) <= suspected({su})."
                )
        if 'fusion_params' not in pkg:
            log.warning(
                "pkg['fusion_params'] absent → paramètres de fusion par défaut utilisés."
            )
        if 'rsf_scaler' not in pkg:
            log.warning("pkg['rsf_scaler'] absent — RSF sans scaling.")

    def _log_thresholds(self) -> None:
        thr = self._get_thresholds()
        thr_src = 'pkg' if 'score_thresholds' in self._pkg else 'défaut'
        log.info(
            f"   Seuils → available≥{thr['available']}  suspected≥{thr['suspected']}  "
            f"(source={thr_src})"
        )
        fp = self._get_fusion_params()
        fusion_src = 'pkg' if 'fusion_params' in self._pkg else 'défaut'
        log.info(
            f"   Fusion → w_xgb_short={fp['w_xgb_short']}  w_rsf_short={fp['w_rsf_short']}  "
            f"w_short={fp['w_short']}  w_long={fp['w_long']}  "
            f"(source={fusion_src})"
        )

    def reload(self) -> Dict[str, str]:
        old_sha     = self._pkg_sha
        old_version = self._pkg.get('version', '?') if self._pkg else '?'
        with self._lock:
            self._pkg     = None
            self._pkg_sha = None
        self._load()
        return {
            'reloaded'    : True,
            'old_version' : old_version,
            'new_version' : self._pkg.get('version', '?'),   # type: ignore[union-attr]
            'old_sha'     : old_sha or 'none',
            'new_sha'     : self._pkg_sha,
            'loaded_at'   : self._loaded_at.isoformat(),     # type: ignore[union-attr]
        }

    def is_stale(self) -> bool:
        pkl_path = CONFIG['MODEL_PKL']
        if not os.path.exists(pkl_path) or self._pkg_sha is None:
            return False
        with open(pkl_path, 'rb') as f:
            current_sha = hashlib.sha256(f.read()).hexdigest()[:12]
        return current_sha != self._pkg_sha

    def _get_thresholds(self) -> Dict[str, int]:
        if self._pkg and 'score_thresholds' in self._pkg:
            thr = self._pkg['score_thresholds']
            return {
                'available': int(thr.get('available', CONFIG['DEFAULT_THR_AVAILABLE'])),
                'suspected': int(thr.get('suspected', CONFIG['DEFAULT_THR_SUSPECTED'])),
            }
        return {
            'available': CONFIG['DEFAULT_THR_AVAILABLE'],
            'suspected': CONFIG['DEFAULT_THR_SUSPECTED'],
        }

    def _get_fusion_params(self) -> Dict[str, float]:
        """Retourne les paramètres de fusion alignés avec compute_scores() de train_model.py"""
        if self._pkg and 'fusion_params' in self._pkg:
            fp = self._pkg['fusion_params']
            return {
                'w_xgb_short'        : float(fp.get('w_xgb_short',        CONFIG['DEFAULT_W_XGB_SHORT'])),
                'w_rsf_short'        : float(fp.get('w_rsf_short',        CONFIG['DEFAULT_W_RSF_SHORT'])),
                'w_xgb_long'         : float(fp.get('w_xgb_long',         CONFIG['DEFAULT_W_XGB_LONG'])),
                'w_rsf_long'         : float(fp.get('w_rsf_long',         CONFIG['DEFAULT_W_RSF_LONG'])),
                'w_short'            : float(fp.get('w_short',             CONFIG['DEFAULT_W_SHORT'])),
                'w_long'             : float(fp.get('w_long',              CONFIG['DEFAULT_W_LONG'])),
                'avail_confident_thr': float(fp.get('avail_confident_thr', CONFIG['DEFAULT_AVAIL_CONFIDENT_THR'])),
                'avail_reduction'    : float(fp.get('avail_reduction',     CONFIG['DEFAULT_AVAIL_REDUCTION'])),
                'na_confident_thr'   : float(fp.get('na_confident_thr',    CONFIG['DEFAULT_NA_CONFIDENT_THR'])),
                'na_confident_w'     : float(fp.get('na_confident_w',      CONFIG['DEFAULT_NA_CONFIDENT_W'])),
                'divergence_thr'     : float(fp.get('divergence_thr',      CONFIG['DEFAULT_DIVERGENCE_THR'])),
                'divergence_w_xgb'   : float(fp.get('divergence_w_xgb',   CONFIG['DEFAULT_DIVERGENCE_W_XGB'])),
                'divergence_w_rsf'   : float(fp.get('divergence_w_rsf',   CONFIG['DEFAULT_DIVERGENCE_W_RSF'])),
                'avail_threshold'    : float(fp.get('avail_threshold',     0.5)),
            }
        return {
            'w_xgb_short'        : CONFIG['DEFAULT_W_XGB_SHORT'],
            'w_rsf_short'        : CONFIG['DEFAULT_W_RSF_SHORT'],
            'w_xgb_long'         : CONFIG['DEFAULT_W_XGB_LONG'],
            'w_rsf_long'         : CONFIG['DEFAULT_W_RSF_LONG'],
            'w_short'            : CONFIG['DEFAULT_W_SHORT'],
            'w_long'             : CONFIG['DEFAULT_W_LONG'],
            'avail_confident_thr': CONFIG['DEFAULT_AVAIL_CONFIDENT_THR'],
            'avail_reduction'    : CONFIG['DEFAULT_AVAIL_REDUCTION'],
            'na_confident_thr'   : CONFIG['DEFAULT_NA_CONFIDENT_THR'],
            'na_confident_w'     : CONFIG['DEFAULT_NA_CONFIDENT_W'],
            'divergence_thr'     : CONFIG['DEFAULT_DIVERGENCE_THR'],
            'divergence_w_xgb'   : CONFIG['DEFAULT_DIVERGENCE_W_XGB'],
            'divergence_w_rsf'   : CONFIG['DEFAULT_DIVERGENCE_W_RSF'],
            'avail_threshold'    : 0.5,
        }

    @property
    def thresholds(self) -> Dict[str, int]:
        return self._get_thresholds()

    @property
    def fusion_params(self) -> Dict[str, float]:
        return self._get_fusion_params()

    @property
    def version(self) -> str:
        return self._pkg.get('version', '?') if self._pkg else '?'

    @property
    def sha(self) -> Optional[str]:
        return self._pkg_sha

    def get_xgb_feature_cols(self) -> List[str]:
        """Retourne les colonnes XGBoost sauvegardées dans le PKL (priorité) ou les defaults."""
        if self._pkg:
            return self._pkg.get('xgb_feature_cols', XGB_FEATURE_COLS)
        return XGB_FEATURE_COLS

    def get_rsf_feature_cols(self) -> List[str]:
        """Retourne les colonnes RSF sauvegardées dans le PKL (priorité) ou les defaults."""
        if self._pkg:
            return self._pkg.get('rsf_feature_cols', RSF_FEATURE_COLS)
        return RSF_FEATURE_COLS

    def get_label_map(self) -> Dict[str, int]:
        if self._pkg:
            return self._pkg.get('label_map', {'Available': 0, 'Suspected': 1, 'NA': 2})
        return {'Available': 0, 'Suspected': 1, 'NA': 2}

    def get_avail_threshold(self) -> float:
        if self._pkg:
            # avail_threshold peut être à la racine du PKL ou dans fusion_params
            thr = self._pkg.get('avail_threshold')
            if thr is not None:
                return float(thr)
            fp = self._pkg.get('fusion_params', {})
            return float(fp.get('avail_threshold', 0.5))
        return 0.5


_registry = _ModelRegistry()


# ══════════════════════════════ VALIDATION DES ENTRÉES ═══════════════════════
class InputValidator:
    KNOWN_KEYS = {'msisdns'}

    def validate(self, body: Any) -> Tuple[bool, List[str], List[str], Dict]:
        errors: List[str] = []
        warnings: List[str] = []
        clean: Dict = {}

        if not isinstance(body, dict):
            return False, ['Le corps de la requête doit être un objet JSON.'], [], {}

        unknown = set(body.keys()) - self.KNOWN_KEYS
        if unknown:
            warnings.append(f"Clés ignorées : {sorted(unknown)}")

        raw_msisdns = body.get('msisdns')
        if raw_msisdns is None:
            errors.append("Champ 'msisdns' obligatoire.")
        elif not isinstance(raw_msisdns, list):
            errors.append("'msisdns' doit être un tableau JSON (liste).")
        elif len(raw_msisdns) == 0:
            errors.append("'msisdns' ne peut pas être vide.")
        elif len(raw_msisdns) > CONFIG['MAX_MSISDNS_PER_REQ']:
            errors.append(
                f"'msisdns' contient {len(raw_msisdns)} éléments "
                f"(max = {CONFIG['MAX_MSISDNS_PER_REQ']})."
            )
        else:
            valid_msisdns, msisdn_errors = self._validate_msisdns(raw_msisdns)
            errors.extend(msisdn_errors)
            clean['msisdns'] = valid_msisdns

        is_valid = len(errors) == 0
        return is_valid, errors, warnings, clean

    @staticmethod
    def _validate_msisdns(raw: List[Any]) -> Tuple[List[str], List[str]]:
        valid: List[str] = []
        errors: List[str] = []
        seen: set = set()
        dup_count = 0

        for idx, item in enumerate(raw):
            if not isinstance(item, (str, int, float)):
                errors.append(
                    f"msisdns[{idx}] : type invalide ({type(item).__name__}), "
                    f"attendu string ou entier."
                )
                continue
            msisdn = str(item).strip()
            if not msisdn:
                errors.append(f"msisdns[{idx}] : valeur vide.")
                continue
            if not _MSISDN_RE.match(msisdn):
                errors.append(
                    f"msisdns[{idx}] = '{msisdn}' : format invalide "
                    f"(attendu : chiffres uniquement, optionnellement préfixé '+', "
                    f"{CONFIG['MSISDN_MIN_LEN']}–{CONFIG['MSISDN_MAX_LEN']} chiffres)."
                )
                continue
            if msisdn in seen:
                dup_count += 1
                continue
            seen.add(msisdn)
            valid.append(msisdn)

        if dup_count > 0:
            errors.append(
                f"{dup_count} MSISDN(s) en doublon supprimé(s) — "
                f"seule la première occurrence est conservée."
            )
        return valid, errors

    @staticmethod
    def validate_features_direct(row: Dict) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for feat in XGB_FEATURE_COLS:
            if feat not in row:
                errors.append(f"Feature manquante : '{feat}'.")
                continue
            val = row[feat]
            if not isinstance(val, (int, float)):
                errors.append(
                    f"'{feat}' : type invalide ({type(val).__name__}), "
                    f"attendu numérique."
                )
                continue
            lo, hi = FEATURE_RANGES[feat]
            if not (lo <= float(val) <= hi):
                errors.append(f"'{feat}' = {val} hors plage [{lo}, {hi}].")
        return len(errors) == 0, errors


_validator = InputValidator()


# ═══════════════════════════════ DONNÉES ═════════════════════════════════════
def _load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _all_progress() -> List[Dict]:
    if CONFIG['USE_MONGODB']:
        from pymongo import MongoClient
        client = MongoClient(CONFIG['MONGO_URI'])
        col    = client[CONFIG['MONGO_DB']][CONFIG['MONGO_COL_CONTACTS']]
        docs   = list(col.find({}, {'_id': 0}))
        client.close()
        return docs
    return _load_jsonl(CONFIG['JSON_USER_CONTACT'])


# ═══════════════════════════ HISTORIQUE ══════════════════════════════════════
def build_history_from_progress(msisdn: str, all_progress: List[Dict]) -> List[Dict]:

    DATE_FMT     = '%Y-%m-%d %H:%M:%S'
    VALID_STATUS = {1, 2, 4, 8}
    CLEAN_STATUS = {0, 16, 34}

    records = [r for r in all_progress if str(r.get('msisdn', '')) == str(msisdn)]
    if not records:
        return []

    history: List[Dict] = []
    seen: set = set()
    for rec in records:
        if 'status_history' in rec:
            events = rec['status_history']
            for sh in events:
                try:
                    dt = datetime.strptime(sh['status_last_updated_at'], DATE_FMT)
                except (KeyError, ValueError, TypeError):
                    continue
                status = sh.get('status')
                if status in CLEAN_STATUS or status not in VALID_STATUS:
                    continue
                key = (status, dt)
                if key in seen:
                    continue
                seen.add(key)
                history.append({'status': status, 'dt': dt})
        else:
            try:
                dt = datetime.strptime(rec['status_last_updated_at'], DATE_FMT)
            except (KeyError, ValueError, TypeError):
                continue
            status = rec.get('status')
            if status in CLEAN_STATUS or status not in VALID_STATUS:
                continue
            key = (status, dt)
            if key in seen:
                continue
            seen.add(key)
            history.append({'status': status, 'dt': dt})

    history.sort(key=lambda x: x['dt'])
    return history


def _normalise_history(history: List[Dict]) -> List[Dict]:
    return [
        {'status': DLR_NORM.get(h['status'], h['status']), 'dt': h['dt']}
        for h in history
    ]

def build_features_from_history(
    msisdn: str,
    history: List[Dict],
    ref_date: Optional[datetime] = None,
) -> Optional[Dict]:

    if ref_date is None:
        ref_date = datetime.now()
    if not history:
        return None

    norm = _normalise_history(history)
    if not norm:
        return None

    # ── Séparation temporelle identique à train_model.py ──────────────────
    last_entry  = norm[-1]
    hist_past   = norm[:-1]
    last_status = last_entry['status']
    last_dt     = last_entry['dt']

    statuses_past = [h['status'] for h in hist_past]
    dts_past      = [h['dt']     for h in hist_past]
    n_hist        = len(hist_past)

    # Counts absolus
    n_envois_hist = n_hist
    n_succes_hist = sum(1 for s in statuses_past if s in DLR_SUCCESS)
    n_echecs_hist = sum(1 for s in statuses_past if s in DLR_FAILURE)

    # Échecs consécutifs max sur historique passé
    echecs_consecutifs_max = 0
    cur = 0
    for s in statuses_past:
        cur = cur + 1 if s in DLR_FAILURE else 0
        echecs_consecutifs_max = max(echecs_consecutifs_max, cur)

    # Échecs consécutifs en fin d'historique passé
    echecs_consecutifs_fin = 0
    for s in reversed(statuses_past):
        if s in DLR_FAILURE:
            echecs_consecutifs_fin += 1
        else:
            break

    # Ancienneté du dernier envoi
    jours_depuis_dernier_envoi = max(0, (ref_date - last_dt).days)

    # Ancienneté du dernier succès dans hist_past
    last_ok_dt = next(
        (h['dt'] for h in reversed(hist_past) if h['status'] in DLR_SUCCESS), None
    )
    jours_depuis_succes = int((ref_date - last_ok_dt).days) if last_ok_dt else 999

    # Score de récence pondéré (time-decay λ=0.02, identique à train_model.py)
    DECAY_LAMBDA = 0.02
    score_recence_pondere = 0.0
    for h in hist_past:
        age = max(0, (ref_date - h['dt']).days)
        w   = np.exp(-DECAY_LAMBDA * age)
        if h['status'] in DLR_SUCCESS:
            score_recence_pondere += w
        elif h['status'] in DLR_FAILURE:
            penalty = 1.5 if age < 7 else 0.8
            score_recence_pondere -= w * penalty
    score_recence_pondere /= max(1.0, np.sqrt(max(1, n_envois_hist)))
    score_recence_pondere = float(np.clip(score_recence_pondere, -10.0, 10.0))

    # Cadence inter-envoi sur hist_past
    if len(dts_past) > 1:
        span = max(1, (dts_past[-1] - dts_past[0]).days)
        freq_inter_envoi_jours = round(span / (len(dts_past) - 1), 2)
    else:
        freq_inter_envoi_jours = 0.0

    # Durée d'observation totale de hist_past
    if len(dts_past) > 1:
        duree_observation_jours = max(1, (dts_past[-1] - dts_past[0]).days)
    elif len(dts_past) == 1:
        duree_observation_jours = max(1, (ref_date - dts_past[0]).days)
    else:
        duree_observation_jours = 1

    cutoff_30j  = last_dt - timedelta(days=30)
    hist_30j    = [h for h in hist_past if h['dt'] >= cutoff_30j]
    n_succes_30j = sum(1 for h in hist_30j if h['status'] in DLR_SUCCESS)
    n_echecs_30j = sum(1 for h in hist_30j if h['status'] in DLR_FAILURE)
    n_total_30j  = len(hist_30j)
    taux_succes_30j = round((n_succes_30j + 1) / (n_total_30j + 2), 4)

    return {
        'msisdn'                    : msisdn,
        'n_envois_hist'             : n_envois_hist,
        'n_succes_hist'             : n_succes_hist,
        'n_echecs_hist'             : n_echecs_hist,
        'echecs_consecutifs_max'    : echecs_consecutifs_max,
        'echecs_consecutifs_fin'    : echecs_consecutifs_fin,
        'jours_depuis_dernier_envoi': jours_depuis_dernier_envoi,
        'jours_depuis_succes'       : jours_depuis_succes,
        'score_recence_pondere'     : round(score_recence_pondere, 4),
        'freq_inter_envoi_jours'    : freq_inter_envoi_jours,
        'duree_observation_jours'   : duree_observation_jours,
        'n_succes_30j'              : n_succes_30j,
        'n_echecs_30j'              : n_echecs_30j,
        'taux_succes_30j'           : taux_succes_30j,
    }


def _default_available(msisdn: str) -> Dict:
    return {
        'msisdn'             : msisdn,
        'decision'           : 'Available',
        'action'             : 'Nouveau contact — aucun historique',
        'availability_score' : 75.0,
        'p_available_%'      : 75.0,
        'p_suspect_%'        : 15.0,
        'p_na_xgb_%'         : 10.0,
        'p_na_rsf_7d_%'      : 5.0,
        'p_na_rsf_30d_%'     : 10.0,
        'model_version'      : _registry.version,
    }


# ═══════════════════════════ RSF FALLBACK ════════════════════════════════════
def _rsf_fallback_single(row: Dict) -> Tuple[float, float]:

    n_hist   = float(row.get('n_envois_hist', 0))
    n_succ   = float(row.get('n_succes_hist', 0))
    ec_fin   = float(row.get('echecs_consecutifs_fin', 0))
    jds      = float(row.get('jours_depuis_succes', 999))
    sr       = float(row.get('score_recence_pondere', 0))
    t30      = float(row.get('taux_succes_30j', 0.5))

    taux = (n_succ / max(1, n_hist))
    risk = (
        (1 - taux)         * 0.30
        + (1 - t30)         * 0.25
        + min(ec_fin / 6.0, 1.0) * 0.20
        + min(jds / 300.0, 1.0)  * 0.15
        + max(0.0, -sr / (abs(sr) + 1)) * 0.10
    )
    risk = float(np.clip(risk, 0.01, 0.97))
    return round(risk * 0.65, 4), round(risk, 4)


# ═══════════════════════ PRÉDICTION : fusion  ══════════════════════════════
def predict_contacts(feature_rows: List[Dict]) -> List[Dict]:

    if not feature_rows:
        return []

    pkg = _registry.pkg
    xgb_model   = pkg['xgboost_model']
    rsf_model   = pkg.get('rsf_model')
    rsf_scaler  = pkg.get('rsf_scaler')
    label_map   = _registry.get_label_map()
    thr         = _registry.thresholds
    fp          = _registry.fusion_params
    avail_thr   = _registry.get_avail_threshold()
    model_ver   = _registry.version

    # Utiliser les colonnes sauvegardées dans le PKL
    xgb_cols = _registry.get_xgb_feature_cols()
    rsf_cols  = _registry.get_rsf_feature_cols()

    # Indices des classes dans label_map (dynamique, aligné avec train_model.py)
    idx_avail   = label_map.get('Available', 0)
    idx_suspect = label_map.get('Suspected', 1)
    idx_na      = label_map.get('NA', 2)
    n_classes   = len(label_map)

    # Matrice features XGBoost
    try:
        X_xgb = np.array([[r[f] for f in xgb_cols] for r in feature_rows], dtype=float)
    except KeyError as e:
        log.error(f"Feature manquante dans feature_rows : {e}")
        raise

    xgb_proba = xgb_model.predict_proba(X_xgb)

    if xgb_proba.shape[1] != n_classes:
        log.warning(
            f"xgb_proba.shape[1]={xgb_proba.shape[1]} ≠ n_classes={n_classes}"
        )
        idx_avail   = min(idx_avail,   xgb_proba.shape[1] - 1)
        idx_suspect = min(idx_suspect, xgb_proba.shape[1] - 1)
        idx_na      = min(idx_na,      xgb_proba.shape[1] - 1)

    # Prédictions RSF
    rsf_preds: Optional[List[Dict]] = None
    rsf_discriminant = False

    if rsf_model is not None:
        try:
            X_rsf = np.array([[r[f] for f in rsf_cols] for r in feature_rows], dtype=float)
            X_in  = rsf_scaler.transform(X_rsf) if rsf_scaler is not None else X_rsf
            surv_funcs = rsf_model.predict_survival_function(X_in)
            t_7, t_30 = 7.0, 30.0
            rsf_preds = []
            for fn in surv_funcs:
                s7  = float(np.interp(t_7,  fn.x, fn.y, left=fn.y[0],  right=fn.y[-1]))
                s30 = float(np.interp(t_30, fn.x, fn.y, left=fn.y[0],  right=fn.y[-1]))
                p7  = float(np.clip(1.0 - s7,  0.0, 1.0))
                p30 = float(np.clip(1.0 - s30, 0.0, 1.0))
                p30 = max(p30, p7)
                rsf_preds.append({'prob_na_7d': round(p7, 4), 'prob_na_30d': round(p30, 4)})

            # Vérification que RSF est discriminant (identique à compute_scores de train_model.py)
            p7_vals  = np.array([r['prob_na_7d']  for r in rsf_preds])
            p30_vals = np.array([r['prob_na_30d'] for r in rsf_preds])
            mean_p30 = float(np.mean(p30_vals))
            rsf_discriminant = bool(
                np.std(p7_vals) > 0.01
                and np.mean(np.abs(p7_vals - p30_vals)) > 0.01
                and mean_p30 < 0.80   # guard : p30 saturé = artefact du cap durée RSF
            )
            if not rsf_discriminant:
                log.warning(
                    f"RSF non discriminant (mean_p30={mean_p30:.3f}) → "
                    f"décision basée sur XGBoost + heuristique temporelle."
                )
        except Exception as exc:
            log.warning(f"RSF predict failed → fallback heuristique : {exc}")
            rsf_preds = None

    results: List[Dict] = []
    for i, row in enumerate(feature_rows):
        p_avail   = float(xgb_proba[i][idx_avail])
        p_suspect = float(xgb_proba[i][idx_suspect])
        p_na_xgb  = float(xgb_proba[i][idx_na])

        if rsf_discriminant and rsf_preds and i < len(rsf_preds):
            p7  = rsf_preds[i]['prob_na_7d']
            p30 = rsf_preds[i]['prob_na_30d']

            # ── Fusion identique à compute_scores() corrigé de train_model.py ──
            na_risk_short  = fp['w_xgb_short'] * p_na_xgb + fp['w_rsf_short'] * p7
            na_risk_long   = fp['w_xgb_long']  * p_na_xgb + fp['w_rsf_long']  * p30
            na_risk_fusion = fp['w_short'] * na_risk_short + fp['w_long'] * na_risk_long

            # Correction si XGBoost très confiant Available
            if p_avail >= fp['avail_confident_thr']:
                na_risk_fusion *= (1.0 - fp['avail_reduction'] * p_avail)
            # Renforcement si XGBoost très confiant NA
            if p_na_xgb >= fp['na_confident_thr']:
                na_risk_fusion = (
                    fp['na_confident_w'] * p_na_xgb
                    + (1 - fp['na_confident_w']) * na_risk_fusion
                )
            # Divergence forte XGB vs RSF → valeur conservative
            if abs(p_na_xgb - p7) > fp['divergence_thr']:
                na_risk_fusion = max(
                    na_risk_fusion,
                    fp['divergence_w_xgb'] * p_na_xgb + fp['divergence_w_rsf'] * p7,
                )

            na_risk = float(np.clip(na_risk_fusion, 0.0, 1.0))
        else:
            # RSF absent/saturé — XGBoost + correctif temporel heuristique
            # (identique au fallback de compute_scores dans train_model.py)
            jds = float(row.get('jours_depuis_succes', 999))
            srp = float(row.get('score_recence_pondere', 0.0))
            ec  = float(row.get('echecs_consecutifs_fin', 0))
            temporal_penalty = (
                float(np.clip(jds  / 999.0, 0.0, 1.0)) * 0.20
                + float(np.clip(-srp / 5.0,  0.0, 1.0)) * 0.10
                + float(np.clip(ec   / 6.0,  0.0, 1.0)) * 0.10
            )
            na_risk = float(np.clip(p_na_xgb + temporal_penalty, 0.0, 1.0))
            p7  = p_na_xgb
            p30 = p_na_xgb

        score = round((1.0 - na_risk) * 100.0, 1)

        # ── Règle de décision identique à compute_scores() de train_model.py ──
        if p_avail >= avail_thr and score >= thr['available']:
            decision, action = 'Available', 'Inclure dans la campagne'
        elif p_avail >= avail_thr and score >= thr['suspected']:
            decision, action = 'Suspected', 'Surveiller — inclure avec prudence'
        elif score >= thr['suspected']:
            decision, action = 'Suspected', 'Surveiller — inclure avec prudence'
        else:
            decision, action = 'NA', 'Exclure de la campagne'

        results.append({
            'msisdn'             : row['msisdn'],
            'decision'           : decision,
            'action'             : action,
            'availability_score' : score,
            'p_available_%'      : round(p_avail   * 100, 1),
            'p_suspect_%'        : round(p_suspect * 100, 1),
            'p_na_xgb_%'         : round(p_na_xgb  * 100, 1),
            'p_na_rsf_7d_%'      : round(p7  * 100, 1),
            'p_na_rsf_30d_%'     : round(p30 * 100, 1),
            'model_version'      : model_ver,
        })

    return results


# ═══════════════════════════════ FLASK APP ═══════════════════════════════════
app = Flask(__name__)
CORS(app)


def _err(msg: str, code: int = 400, **extra) -> Tuple:
    body = {'error': msg, **extra}
    return jsonify(body), code


def _validation_error(errors: List[str], warnings: List[str]) -> Tuple:
    body: Dict[str, Any] = {
        'error'  : 'Validation des entrées échouée.',
        'details': errors,
    }
    if warnings:
        body['warnings'] = warnings
    return jsonify(body), 422


# ─────────────────────── POST /api/predict ───────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    started = datetime.now()

    body = request.get_json(silent=True)
    if body is None:
        return _err(
            "Corps de requête invalide ou Content-Type manquant "
            "(attendu : application/json).",
            400,
        )

    is_valid, errors, warnings, clean = _validator.validate(body)
    if not is_valid:
        log.warning(f"POST /api/predict — validation échouée : {errors}")
        return _validation_error(errors, warnings)

    msisdns_input = clean['msisdns']
    log.info(f"POST /api/predict  msisdns={len(msisdns_input)}")
    if warnings:
        log.warning(f"Validation warnings : {warnings}")

    try:
        _ = _registry.pkg
    except FileNotFoundError as exc:
        return _err(str(exc), 503)

    if _registry.is_stale():
        log.warning(
            "Le fichier .pkl a changé sur disque depuis le dernier chargement. "
            "Appelez POST /api/model/reload pour recharger."
        )

    all_progress = _all_progress()
    rows_hist: List[Dict] = []
    rows_none: List[str]  = []

    for msisdn in msisdns_input:
        history = build_history_from_progress(msisdn, all_progress)
        if history:
            row = build_features_from_history(msisdn, history)
            if row:
                rows_hist.append(row)
            else:
                rows_none.append(msisdn)
        else:
            rows_none.append(msisdn)

    predicted = predict_contacts(rows_hist)
    for m in rows_none:
        predicted.append(_default_available(m))

    elapsed = round((datetime.now() - started).total_seconds() * 1000)

    summary = {
        'total'    : len(predicted),
        'available': sum(1 for r in predicted if r['decision'] == 'Available'),
        'suspected': sum(1 for r in predicted if r['decision'] == 'Suspected'),
        'na'       : sum(1 for r in predicted if r['decision'] == 'NA'),
    }
    log.info(
        f"→ total={summary['total']}  avail={summary['available']}  "
        f"susp={summary['suspected']}  na={summary['na']}  ({elapsed}ms)"
    )

    resp: Dict[str, Any] = {
        'total'             : len(predicted),
        'contacts'          : predicted,
        'elapsed_ms'        : elapsed,
        'model_version'     : _registry.version,
        'model_sha'         : _registry.sha,
        'predicted_at'      : datetime.now().isoformat(),
        'active_thresholds' : _registry.thresholds,
        'active_fusion'     : _registry.fusion_params,
    }
    if warnings:
        resp['input_warnings'] = warnings

    return jsonify(resp), 200


# ─────────────────────── POST /api/predict_direct ────────────────────────────
@app.route('/api/predict_direct', methods=['POST'])
def predict_direct():

    body = request.get_json(silent=True)
    if not isinstance(body, dict) or 'contacts' not in body:
        return _err("Champ 'contacts' obligatoire (liste de vecteurs de features).")

    raw_contacts = body['contacts']
    if not isinstance(raw_contacts, list) or len(raw_contacts) == 0:
        return _err("'contacts' doit être une liste non-vide.")

    validated_rows: List[Dict] = []
    all_errors: List[Dict] = []

    for idx, row in enumerate(raw_contacts):
        if not isinstance(row, dict):
            all_errors.append({'index': idx, 'errors': ['Doit être un objet JSON.']})
            continue
        ok, errs = InputValidator.validate_features_direct(row)
        if not ok:
            all_errors.append({'index': idx, 'msisdn': row.get('msisdn', '?'), 'errors': errs})
        else:
            if 'msisdn' not in row:
                row['msisdn'] = f'__direct_{idx}'
            validated_rows.append(row)

    if all_errors:
        return jsonify({
            'error'  : 'Validation des features échouée.',
            'details': all_errors,
        }), 422

    try:
        _ = _registry.pkg
    except FileNotFoundError as exc:
        return _err(str(exc), 503)

    results = predict_contacts(validated_rows)
    return jsonify({'contacts': results, 'model_version': _registry.version}), 200


# ─────────────────────── GET /api/health ─────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    try:
        pkg = _registry.pkg
        return jsonify({
            'status'               : 'ok',
            'model_version'        : _registry.version,
            'model_sha'            : _registry.sha,
            'trained_at'           : pkg.get('trained_at', '?'),
            'loaded_at'            : _registry.loaded_at.isoformat() if _registry.loaded_at else None,
            'fusion_logic'         : pkg.get('fusion_logic', '?'),
            'n_features_xgb'       : len(_registry.get_xgb_feature_cols()),
            'n_features_rsf'       : len(_registry.get_rsf_feature_cols()),
            'xgb_feature_cols'     : _registry.get_xgb_feature_cols(),
            'rsf_scaler'           : 'present' if pkg.get('rsf_scaler') is not None else 'absent',
            'label_map'            : _registry.get_label_map(),
            'avail_threshold'      : _registry.get_avail_threshold(),
            'score_thresholds'     : _registry.thresholds,
            'thresholds_source'    : 'pkg' if 'score_thresholds' in pkg else 'default',
            'fusion_params'        : _registry.fusion_params,
            'fusion_params_source' : 'pkg' if 'fusion_params' in pkg else 'default',
            'is_stale'             : _registry.is_stale(),
        }), 200
    except FileNotFoundError as exc:
        return jsonify({'status': 'error', 'detail': str(exc)}), 503


# ─────────────────────── POST /api/model/reload ──────────────────────────────
@app.route('/api/model/reload', methods=['POST'])
def model_reload():
    log.info("POST /api/model/reload — rechargement demandé")
    try:
        info = _registry.reload()
        log.info(
            f"Modèle rechargé : {info['old_version']} → {info['new_version']}  "
            f"sha: {info['old_sha']} → {info['new_sha']}"
        )
        return jsonify(info), 200
    except FileNotFoundError as exc:
        return _err(str(exc), 503)
    except Exception as exc:
        log.error(f"Reload échoué : {exc}", exc_info=True)
        return _err(f"Reload échoué : {exc}", 500)


# ─────────────────────── GET /api/contacts ───────────────────────────────────
@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    contacts_path = str(_BASE_DIR / 'contacts.json')
    try:
        with open(contacts_path, encoding='utf-8') as f:
            all_contacts = json.load(f)
    except FileNotFoundError:
        return _err('contacts.json introuvable', 404)

    contacts = list(all_contacts)
    tag    = request.args.get('tag')
    tag_id = request.args.get('tag_id')
    if tag:
        contacts = [c for c in contacts if c.get('tag', '').lower() == tag.lower()]
    elif tag_id:
        contacts = [c for c in contacts if str(c.get('tag_id', '')) == str(tag_id)]

    tags: Dict[str, int] = {}
    for c in all_contacts:
        t = c.get('tag', '')
        if t:
            tags[t] = tags.get(t, 0) + 1

    return jsonify({
        'total'   : len(contacts),
        'contacts': contacts,
        'tags'    : [{'name': k, 'count': v} for k, v in tags.items()],
    }), 200


# ═══════════════════════════ POINT D'ENTRÉE ══════════════════════════════════
if __name__ == '__main__':
    log.info("  Contact Availability API »")
    log.info("=" * 60)
    log.info(f"  Mode    : {'MongoDB' if CONFIG['USE_MONGODB'] else 'JSON'}")
    log.info(f"  Écoute  : http://{CONFIG['HOST']}:{CONFIG['PORT']}")
    log.info(f"  Features XGB ({len(XGB_FEATURE_COLS)}) : {XGB_FEATURE_COLS}")
    log.info(f"  Features RSF ({len(RSF_FEATURE_COLS)}) : {RSF_FEATURE_COLS}")
    log.info("-" * 60)
    try:
        _registry.pkg
        log.info("  Modèle pré-chargé au démarrage ")
    except FileNotFoundError as exc:
        log.warning(f"  Modèle absent au démarrage — sera chargé au 1er appel : {exc}")
    app.run(host=CONFIG['HOST'], port=CONFIG['PORT'], debug=CONFIG['DEBUG'])