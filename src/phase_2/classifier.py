import ollama, json, re, time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import LLM

SYSTEM_PROMPT = """/no_think
You are classifying a Food Sharing Initiative (FSI) for an academic dataset.
Respond ONLY with a valid JSON object — no prose, no markdown, no explanation.

The text may be in English, Spanish, Catalan, Italian, or another language. Classify based
on meaning regardless of language. Always respond in English with the exact values below.

Key terms to recognise across languages:
- hort / huerto / orto = urban garden → community_garden
- cooperativa de consum / grupo de consumo / cooperativa di consumo = buying cooperative → food_sharing
- banc dels aliments / banco de alimentos / banca del cibo = food bank → food_bank
- menjador social / comedor social / mensa sociale = social dining → meals_service
- intercanvi / intercambio / scambio = exchange/swapping → food_swapping
- nevera solidaria / nevera solidàría / frigo solidale = solidarity fridge → food_gifting
- fundació / fundación / fondazione + food aid = ngo_led

The object must have exactly these two keys:
- fsi_type: one of [food_sharing, food_swapping, food_gifting, community_garden,
                    food_bank, meals_service, food_education, other]
- operational_level: one of [government_funded, council_supported, ngo_led,
                              community_led, commercial, unknown]

If the text is clearly insufficient (e.g. only navigation links or an error page), use "unknown".
But if ANY food-related keywords are present, make your best guess rather than returning "unknown"."""

_OP_RULES = [
    (["government funded", "government grant", "government support",
      "department of", "hse funded", "state funded", "state grant",
      "public funding", "exchequer"], "government_funded"),
    (["city council", "county council", "dublin city council",
      "council funded", "council grant", "council support",
      "local authority", "local government",
      "ajuntament", "ayuntamiento", "comune di"], "council_supported"),
    (["registered charity", "charity number", "ngo", "non-profit",
      "nonprofit", "charitable organisation", "charitable organization",
      "foundation grant", "funded by a charity",
      "fundació", "fundación", "fondazione"], "ngo_led"),
    (["community group", "community-led", "community led",
      "grassroots", "volunteer-run", "volunteer run",
      "run by volunteers", "community initiative",
      "neighbourhood group", "neighbors", "neighbours",
      "associació de veïns", "associazione di quartiere"], "community_led"),
    (["social enterprise", "community interest company", "cic",
      "trading", "revenue", "membership fee", "subscription fee",
      "pay as you feel"], "commercial"),
]

_KEYWORD_RULES = [
    (["hort", "huerto", "orto", "jardí", "jardi", "jardín", "jardin",
      "urban garden", "community garden", "horta", "huertas", "vinyes", "vinya",
      "jardí comunitari", "jardín comunitario", "allotment", "growing space",
      "growing project", "food garden", "garden project"], "community_garden"),
    (["cooperativa de consum", "grupo de consumo", "consum responsable",
      "compra colectiva", "compra col·lectiva", "grup de consum",
      "grupo de consumidores", "queviure",
      "food sharing", "share food", "shared food", "redistribut",
      "surplus food"], "food_sharing"),
    (["banc dels aliments", "banco de alimentos", "banca del cibo",
      "food bank", "food pantry", "larder", "food hub", "foodbank",
      "banc d'aliments", "emergency food"], "food_bank"),
    (["nevera solid", "frigo solid", "solidarity fridge", "community fridge",
      "aprofitem els aliments", "aprofitem", "malbaratament",
      "rescat d'aliments", "rescate de alimentos", "aliments gratuïts",
      "food gifting", "free food", "take what you need"], "food_gifting"),
    (["menjador social", "comedor social", "mensa sociale",
      "meals service", "soup kitchen", "cocina social",
      "servei de menjador", "hot meal", "free meals", "community meal"], "meals_service"),
    (["intercanvi", "intercambio", "scambio", "food swap",
      "food swapping", "swap shop", "swapping", "trueque"], "food_swapping"),
    (["educació alimentària", "educación alimentaria", "educazione alimentare",
      "food education", "taller d'alimentació", "workshop", "formació alimentà",
      "compostatge", "compostaje", "agricultura ecològica", "permacultura",
      "cooking class", "cooking course", "nutrition workshop",
      "food literacy"], "food_education"),
]


def _keyword_fallback(record: dict) -> str | None:
    combined = (
        (record.get("url") or "") + " " +
        (record.get("title") or "") + " " +
        (record.get("description") or "") + " " +
        (record.get("text") or "")[:500]
    ).lower()
    for keywords, fsi_type in _KEYWORD_RULES:
        if any(kw in combined for kw in keywords):
            return fsi_type
    return None


def _op_fallback(record: dict) -> str | None:
    combined = (
        (record.get("title") or "") + " " +
        (record.get("description") or "") + " " +
        (record.get("text") or "")[:800]
    ).lower()
    for keywords, op_level in _OP_RULES:
        if any(kw in combined for kw in keywords):
            return op_level
    return None


def _build_user_prompt(batch: list[dict]) -> str:
    r            = batch[0]
    desc         = (r.get("description") or "")[:200]
    text_preview = (r.get("text") or "")[:800]
    content      = f"{desc}\n{text_preview}".strip()[:1000]
    return f"Classify this FSI:\n\n{r['title']}\n{content}"


_DECODER = json.JSONDecoder()


def _extract_json(raw: str) -> list | dict:
    raw = raw.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = raw.find("[")
    if start != -1:
        try:
            value, _ = _DECODER.raw_decode(raw[start:])
            return value
        except json.JSONDecodeError:
            pass
    start = raw.find("{")
    if start != -1:
        try:
            value, _ = _DECODER.raw_decode(raw[start:])
            return [value]
        except json.JSONDecodeError:
            pass
    obj_strs = re.findall(r'\{[^{}]+\}', raw)
    if obj_strs:
        return [json.loads(s) for s in obj_strs]
    raise ValueError(f"No JSON found in response: {raw[:120]!r}")


def _parse_response(raw: str, expected_count: int) -> list[dict]:
    results = _extract_json(raw)
    if not isinstance(results, list):
        raise ValueError(f"Response was not a list: {type(results)}")
    if len(results) > expected_count:
        return results[:expected_count]
    padded = len(results)
    while len(results) < expected_count:
        results.append({"fsi_type": "unknown", "operational_level": "unknown"})
    if padded < expected_count:
        print(f"    WARNING: LLM returned {padded} results for batch of "
              f"{expected_count} — padded {expected_count - padded} with 'unknown'")
    return results


def _classify_batch(batch: list[dict], retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            resp = ollama.chat(
                model=LLM["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(batch)},
                ],
                options={"temperature": 0},
            )
            raw = resp["message"]["content"]
            return _parse_response(raw, len(batch))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    Parse error on attempt {attempt+1}: {e} — retrying...")
            time.sleep(1)
        except Exception as e:
            if attempt == retries - 1:
                print(f"    Failed after {retries} attempts: {e}")
            time.sleep(2 ** attempt)

    return [{"fsi_type": "unknown", "operational_level": "unknown"}
            for _ in batch]


def classify_all(records: list[dict]) -> list[dict]:
    to_classify, empties = [], []
    for r in records:
        if r.get("text") or r.get("title"):
            to_classify.append(r)
        else:
            empties.append(r)

    for r in empties:
        r["fsi_type"]          = "unknown"
        r["operational_level"] = "unknown"

    batch_size = LLM["batch_size"]
    batches    = [to_classify[i:i+batch_size]
                  for i in range(0, len(to_classify), batch_size)]

    for i, batch in enumerate(batches):
        print(f"  Classifying batch {i+1}/{len(batches)}...")
        results = _classify_batch(batch)

        for j, record in enumerate(batch):
            classification = results[j] if j < len(results) else {}
            if isinstance(classification, dict):
                record["fsi_type"]          = classification.get("fsi_type", "unknown")
                record["operational_level"] = classification.get("operational_level", "unknown")
            else:
                record["fsi_type"]          = "unknown"
                record["operational_level"] = "unknown"

            if record["fsi_type"] == "unknown":
                fallback = _keyword_fallback(record)
                if fallback:
                    record["fsi_type"] = fallback
            if record["operational_level"] == "unknown":
                fallback = _op_fallback(record)
                if fallback:
                    record["operational_level"] = fallback

    return records
