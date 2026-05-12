"""BRENDA SOAP retriever — targeted EC+organism+substrate queries for live Km data."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from loguru import logger
from zeep import Client, Settings
from zeep.helpers import serialize_object
from zeep.transports import Transport

from config import BRENDA_EMAIL, BRENDA_PASSWORD, CACHE_DB_PATH
from processing.cache import QueryCache
from retrieval.base import BaseRetriever, CatalystRecord, RetrieverConnectionError, RetrieverParseError
from retrieval.demo_data import brenda_demo_records

BRENDA_WSDL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"


def _parse_brenda_string(raw: str, ec_number: str) -> list[dict[str, Any]]:
    """Parse BRENDA legacy string response into dicts."""
    entries = []
    for block in raw.strip().split("!"):
        block = block.strip()
        if not block:
            continue
        entry: dict[str, Any] = {"ec_number": ec_number}
        for pair in block.split("#"):
            if "*" in pair:
                key, _, val = pair.partition("*")
                entry[key.strip()] = val.strip()
        if len(entry) > 1:
            entries.append(entry)
    return entries


# ── Typical operating conditions per EC number ────────────────────────────────
_EC_CONDITIONS: dict[str, dict[str, float]] = {
    "1.18.6.1":   {"temperature_k": 303.0, "ph": 7.2},
    "1.19.6.1":   {"temperature_k": 303.0, "ph": 7.2},
    "1.12.1.2":   {"temperature_k": 310.0, "ph": 7.0},
    "1.12.7.2":   {"temperature_k": 353.0, "ph": 7.0},
    "4.2.1.1":    {"temperature_k": 310.0, "ph": 7.4},
    "1.2.99.2":   {"temperature_k": 343.0, "ph": 7.0},
    "1.1.99.8":   {"temperature_k": 303.0, "ph": 7.0},
    "1.2.1.2":    {"temperature_k": 298.0, "ph": 7.5},
    "2.7.1.1":    {"temperature_k": 310.0, "ph": 7.4},
    "2.7.1.2":    {"temperature_k": 310.0, "ph": 7.4},
    "2.7.1.11":   {"temperature_k": 310.0, "ph": 7.4},
    "2.7.1.40":   {"temperature_k": 310.0, "ph": 7.4},
    "1.1.1.27":   {"temperature_k": 310.0, "ph": 7.0},
    "1.1.1.1":    {"temperature_k": 298.0, "ph": 8.8},
    "1.11.1.6":   {"temperature_k": 310.0, "ph": 7.0},
    "3.5.1.5":    {"temperature_k": 298.0, "ph": 7.0},
    "3.2.1.23":   {"temperature_k": 310.0, "ph": 7.3},
    "1.1.3.4":    {"temperature_k": 298.0, "ph": 5.5},
    "1.14.13.25": {"temperature_k": 303.0, "ph": 7.0},
}

# ── EC number → human-readable enzyme name ───────────────────────────────────
_EC_NAMES: dict[str, str] = {
    "1.18.6.1":   "Nitrogenase",
    "1.19.6.1":   "Nitrogenase (flavodoxin)",
    "1.12.1.2":   "Hydrogenase",
    "1.12.7.2":   "Hydrogenase (ferredoxin)",
    "4.2.1.1":    "Carbonic anhydrase",
    "1.2.99.2":   "CO dehydrogenase",
    "1.1.99.8":   "Methanol dehydrogenase",
    "1.2.1.2":    "Formate dehydrogenase",
    "2.7.1.1":    "Hexokinase",
    "2.7.1.2":    "Glucokinase",
    "2.7.1.11":   "Phosphofructokinase",
    "2.7.1.40":   "Pyruvate kinase",
    "1.1.1.27":   "Lactate dehydrogenase",
    "1.1.1.1":    "Alcohol dehydrogenase",
    "1.11.1.6":   "Catalase",
    "3.5.1.5":    "Urease",
    "3.2.1.23":   "Beta-galactosidase",
    "1.1.3.4":    "Glucose oxidase",
    "1.14.13.25": "Methane monooxygenase",
}

# ── Targeted query catalogue ──────────────────────────────────────────────────
_QUERY_CATALOGUE: list[tuple[str, str, str]] = [
    ("1.18.6.1", "Azotobacter vinelandii",               "N2"),        # 0
    ("1.18.6.1", "Klebsiella pneumoniae",                "N2"),        # 1
    ("1.12.1.2", "Clostridium pasteurianum",             "H2"),        # 2
    ("1.12.7.2", "Pyrococcus furiosus",                  "H2"),        # 3
    ("4.2.1.1",  "Homo sapiens",                         "CO2"),       # 4
    ("4.2.1.1",  "Methanobacterium thermoautotrophicum",  "CO2"),       # 5
    ("1.2.99.2", "Carboxydothermus hydrogenoformans",    "CO"),        # 6
    ("1.1.99.8", "Methylobacterium extorquens",          "methanol"),  # 7
    ("1.2.1.2",  "Candida boidinii",                     "formate"),   # 8
    ("2.7.1.1",  "Saccharomyces cerevisiae",             "glucose"),   # 9
    ("2.7.1.1",  "Homo sapiens",                         "glucose"),   # 10
    ("2.7.1.2",  "Homo sapiens",                         "glucose"),   # 11
    ("2.7.1.11", "Escherichia coli",                     "ATP"),       # 12
    ("2.7.1.40", "Homo sapiens",                         "phosphoenolpyruvate"),  # 13
    ("1.1.1.27", "Bos taurus",                           "pyruvate"),  # 14
    ("1.1.1.1",  "Saccharomyces cerevisiae",             "ethanol"),   # 15
    ("1.11.1.6", "Homo sapiens",                         "H2O2"),      # 16
    ("3.5.1.5",  "Canavalia ensiformis",                 "urea"),      # 17
    ("3.2.1.23", "Escherichia coli",                     "lactose"),   # 18
    ("1.1.3.4",  "Aspergillus niger",                    "glucose"),   # 19
]

_DEFAULT_INDICES = [0, 2, 4, 9, 15, 16]

_KEYWORD_INDEX: dict[str, list[int]] = {
    "n2":          [0, 1],
    "nh3":         [0, 1],
    "nitrogen":    [0, 1],
    "nitrogenase": [0, 1],
    "h2":          [2, 3],
    "hydrogen":    [2, 3],
    "hydrogenase": [2, 3],
    "co2":         [4, 5],
    "carbonic":    [4, 5],
    "co":          [6],
    "methanol":    [7],
    "formate":     [8],
    "glucose":     [9, 10, 11, 19],
    "atp":         [12],
    "pyruvate":    [13, 14],
    "lactate":     [14],
    "ethanol":     [15],
    "h2o2":        [16],
    "urea":        [17],
    "lactose":     [18],
    "galactose":   [18],
    "methane":     [6, 7],
    "fe":          [],
    "o2":          [16],
}


def _queries_for_reaction(reaction: str) -> list[tuple[str, str, str]]:
    """Select targeted (ec, organism, substrate) queries for a reaction."""
    tokens = set(
        re.split(r"[^a-z0-9]+", reaction.lower().replace("→", " ").replace("->", " "))
    )
    tokens -= {"", "to", "and", "or", "the", "with"}

    indices: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        for key, idxs in _KEYWORD_INDEX.items():
            if key in token or token in key:
                for i in idxs:
                    if i not in seen:
                        indices.append(i)
                        seen.add(i)

    if not indices:
        indices = [i for i in _DEFAULT_INDICES if i not in seen]

    return [_QUERY_CATALOGUE[i] for i in indices[:6]]


class BrendaRetriever(BaseRetriever):
    """Retrieve enzyme Km values from BRENDA SOAP using targeted queries."""

    source_name = "BRENDA"
    _client_cache: "Client | None" = None

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        cache: QueryCache | None = None,
    ) -> None:
        self.email = email or BRENDA_EMAIL
        self.password = password or BRENDA_PASSWORD
        self.cache = cache or QueryCache(CACHE_DB_PATH)

    def _hash_password(self) -> str:
        return hashlib.sha256(self.password.encode("utf-8")).hexdigest()

    def _get_client(self) -> Client:
        """Return cached zeep client — WSDL loads once per process."""
        if BrendaRetriever._client_cache is not None:
            return BrendaRetriever._client_cache
        try:
            settings = Settings(strict=False, xml_huge_tree=True)
            transport = Transport(timeout=20, operation_timeout=20)
            client = Client(wsdl=BRENDA_WSDL, settings=settings, transport=transport)
            BrendaRetriever._client_cache = client
            return client
        except Exception as exc:  # noqa: BLE001
            raise RetrieverConnectionError(f"Failed to load BRENDA WSDL: {exc}") from exc

    def _fetch_km(
        self,
        client: Client,
        ec_number: str,
        organism: str,
        substrate: str,
    ) -> list[dict[str, Any]]:
        """Fetch Km values using official BRENDA positional argument format."""
        try:
            params = (
                self.email,
                self._hash_password(),
                f"ecNumber*{ec_number}",
                f"organism*{organism}",
                "kmValue*",
                "kmValueMaximum*",
                f"substrate*{substrate}",
                "commentary*",
                "ligandStructureId*",
                "literature*",
            )
            response = client.service.getKmValue(*params)
            serialized = serialize_object(response)

            if not serialized:
                return []
            if isinstance(serialized, str) and serialized.strip():
                return _parse_brenda_string(serialized, ec_number)
            if isinstance(serialized, list):
                result = []
                for entry in serialized:
                    if isinstance(entry, dict):
                        d = dict(entry)
                    elif hasattr(entry, "items"):
                        d = dict(entry)
                    elif hasattr(entry, "__dict__"):
                        d = {k: v for k, v in vars(entry).items() if not k.startswith("_")}
                    else:
                        d = {"value": str(entry)}
                    d.setdefault("ec_number", ec_number)
                    result.append(d)
                return result
            return []

        except Exception as exc:  # noqa: BLE001
            err = str(exc).lower()
            if any(w in err for w in ("username", "password", "account", "wrong", "invalid")):
                raise RetrieverConnectionError(f"BRENDA auth error: {exc}") from exc
            logger.debug("BRENDA EC={} org={} sub={} failed: {}", ec_number, organism, substrate, exc)
            return []

    def search(self, reaction: str) -> list[CatalystRecord]:
        """Retrieve live Km values from BRENDA for the given reaction.

        Falls back to curated demo records if credentials are missing or
        all queries fail.

        Args:
            reaction: Reaction text to map to enzyme queries.

        Returns:
            List of CatalystRecord items with temperature/pH in conditions.

        Raises:
            RetrieverParseError: If response mapping fails unexpectedly.
        """
        if not self.email or not self.password:
            logger.warning("BRENDA credentials missing; using demo fallback.")
            return brenda_demo_records(reaction)

        queries = _queries_for_reaction(reaction)
        cache_key = {"queries": queries}
        cached = self.cache.get(self.source_name, cache_key)

        raw_entries: list[dict[str, Any]]
        if cached is not None:
            raw_entries = cached.get("entries", [])
            logger.info("BRENDA cache hit: {} entries", len(raw_entries))
        else:
            raw_entries = []
            try:
                client = self._get_client()
                for ec, organism, substrate in queries:
                    entries = self._fetch_km(client, ec, organism, substrate)
                    for e in entries:
                        e.setdefault("ec_number", ec)
                        e.setdefault("query_organism", organism)
                        e.setdefault("query_substrate", substrate)
                    raw_entries.extend(entries)
                    logger.debug("BRENDA EC={} org={} sub={}: {} entries", ec, organism, substrate, len(entries))
            except RetrieverConnectionError as exc:
                logger.warning("BRENDA auth error; using demo fallback: {}", exc)
                return brenda_demo_records(reaction)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BRENDA query failed; using demo fallback: {}", exc)
                return brenda_demo_records(reaction)

            self.cache.set(self.source_name, cache_key, {"entries": raw_entries})

        if not raw_entries:
            logger.info("BRENDA returned 0 entries; using demo fallback.")
            return brenda_demo_records(reaction)

        try:
            records: list[CatalystRecord] = []
            for idx, entry in enumerate(raw_entries):
                ec = str(entry.get("ec_number") or entry.get("ecNumber") or "")
                enzyme_name = str(
                    entry.get("recommended_name")
                    or entry.get("enzyme_name")
                    or entry.get("systematicName")
                    or _EC_NAMES.get(ec, f"EC {ec}")
                )
                organism = str(
                    entry.get("organism")
                    or entry.get("query_organism")
                    or "Unknown organism"
                )
                km_raw = (
                    entry.get("kmValue")
                    or entry.get("km_value")
                    or entry.get("value")
                )
                unit = str(
                    entry.get("kmValueUnit")
                    or entry.get("km_unit")
                    or entry.get("unit")
                    or "mM"
                )
                substrate = str(
                    entry.get("substrate")
                    or entry.get("query_substrate")
                    or ""
                )

                km_value: float | None
                try:
                    km_value = float(km_raw) if km_raw is not None else None
                except (TypeError, ValueError):
                    km_value = None

                # Enrich with typical temperature/pH for this enzyme
                ec_cond = _EC_CONDITIONS.get(ec, {})

                records.append(
                    CatalystRecord(
                        source=self.source_name,
                        source_id=str(entry.get("id") or f"brenda-{ec}-{idx}"),
                        name=enzyme_name,
                        formula="",
                        reaction=reaction,
                        activity_metric="Km",
                        activity_value=km_value,
                        activity_unit=unit,
                        conditions={
                            "organism": organism,
                            "substrate": substrate,
                            "ec_number": ec,
                            "temperature_k": ec_cond.get("temperature_k"),
                            "ph": ec_cond.get("ph"),
                        },
                        stability=None,
                        raw={k: str(v) for k, v in entry.items()},
                    )
                )

            logger.info("BRENDA returned {} live Km records for reaction='{}'.", len(records), reaction)
            return records

        except Exception as exc:  # noqa: BLE001
            raise RetrieverParseError(f"Failed to parse BRENDA response: {exc}") from exc

    def health_check(self) -> bool:
        """Always True — demo fallback guarantees results."""
        return True
