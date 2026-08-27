import ollama, re


def _synthesize(prompt: str, model: str, num_ctx: int) -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": num_ctx},
    )
    text = resp["message"]["content"].strip()
    # strip Qwen3 thinking blocks — let the model reason, discard the scratchpad
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _check_numbers(text: str, allowed_numbers: set[str], section: str):
    """Warn if the prose contains numbers not present in the facts dict."""
    found = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
    hallucinated = found - allowed_numbers
    if hallucinated:
        print(f"  WARNING [{section}]: prose contains numbers not in facts: {hallucinated}")


def synthesize_all(facts: dict, graph_answers: dict, cfg: dict,
                   city: str, country: str, language: str = "en") -> dict:
    model   = cfg["model"]
    num_ctx = cfg["num_ctx"]
    total   = facts["total"]

    def write(instruction: str, data_points: str) -> str:
        prompt = f"""You are writing one section (two short paragraphs) for an academic report about Food Sharing Initiatives (FSIs) in {city}, {country}.

STRICT RULES:
- Write in English regardless of the language of any source material.
- Write exactly 8-10 sentences total, split into two paragraphs.
- First paragraph: cover the headline numbers and main patterns (3-4 sentences).
- Second paragraph: go deeper — compare sub-groups, note contrasts, mention specific examples or implications where the data supports it (4-5 sentences).
- Every number or percentage in your text MUST come from DATA POINTS below — do not invent or estimate any figures.
- You may use ADDITIONAL CONTEXT for qualitative descriptions (what types of FSIs exist, what they do) but never use it as a source for numbers or to introduce new factual claims not supported by DATA POINTS.
- Do not name specific organisations unless their name appears in the data points.
- For each number, explain what it means in context (e.g. not just "15 FSIs are in Dublin 2" but "Dublin 2 hosts 15 FSIs, the highest concentration of any district").
- Be factual, analytical, and professional — this is an academic report.
- No bullet points, no headers, no markdown.
- Do not repeat the section title.
- Do not end with vague closing phrases like "Overall, these findings suggest..." or "This highlights the complexity of...". Every sentence must contain a specific fact or comparison from the data points.

DATA POINTS (the only permitted source for numbers and specific claims):
{data_points}

ADDITIONAL CONTEXT (qualitative background only — do not introduce new factual claims from this):
{instruction}

Write the two paragraphs now:"""
        return _synthesize(prompt, model, num_ctx)

    # build set of all numbers that appear in the facts dict (permitted in prose)
    allowed_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', str(facts)))

    def checked_write(instruction: str, data_points: str, section: str) -> str:
        text = write(instruction, data_points)
        _check_numbers(text, allowed_numbers, section)
        return text

    print("  Synthesizing sections from extracted facts...")
    prose = {}

    # ── Geographic distribution ───────────────────────────────────────────────
    top    = ", ".join(f"{d} ({n} FSIs)" for d, n in facts["top_districts"])
    sparse = ", ".join(facts["sparse_districts"]) or "none identified"
    not_specified = facts["district_counts"].get("Not Specified", 0)

    # combine global distribution answer with per-district qualitative context
    geo_context = graph_answers.get("geographic_distribution", {}).get("answer", "")
    district_detail = graph_answers.get("district_summaries", {}).get("answer", "")
    if district_detail:
        geo_context = geo_context + "\n\nDistrict-level detail:\n" + district_detail[:600]

    sparse_count = len(facts["sparse_districts"])
    all_districts = ", ".join(
        f"{d} ({n})" for d, n in sorted(
            facts["district_counts"].items(), key=lambda x: -x[1]
        ) if d != "Not Specified" and n > 0
    )

    # compute density categories from district counts
    _dc = {d: n for d, n in facts["district_counts"].items()
           if d != "Not Specified" and n > 0}
    if _dc:
        _vals = sorted(_dc.values())
        _n = len(_vals)
        _hi = _vals[max(0, int(0.75 * _n))]
        _lo = _vals[max(0, int(0.25 * _n))]
        green_ds  = [d for d, c in _dc.items() if c >= _hi]
        yellow_ds = [d for d, c in _dc.items() if _lo < c < _hi]
        red_ds    = [d for d, c in _dc.items() if c <= _lo]
    else:
        green_ds = yellow_ds = red_ds = []

    prose["geographic"] = checked_write(
        geo_context,
        f"""- Total FSIs: {total} across {facts['district_count']} postal districts in {city}, {country}
- Full district breakdown (name, count): {all_districts}
- Top 3 districts by concentration are the first three in the breakdown above
- {sparse_count} districts have only 1 FSI each (the lowest-count entries above)
- Green (high-density) districts: {', '.join(green_ds) or 'none'}
- Yellow (medium-density) districts: {', '.join(yellow_ds) or 'none'}
- Red (low-density) districts: {', '.join(red_ds) or 'none'}
- All district assignments use official postcode boundary polygons and GPS coordinates
- Data gap: FSI density per capita requires census population data""",
        "geographic",
    )
    print("    ✓ geographic distribution")

    # ── FSI types ─────────────────────────────────────────────────────────────
    types = "; ".join(
        f"{t}: {n} ({facts['type_pct'][t]}%)"
        for t, n in facts["type_counts"].items()
    )
    prose["types"] = checked_write(
        graph_answers.get("fsi_types", {}).get("answer", ""),
        f"""- Total FSIs: {total}
- Type breakdown: {types}
- FSIs with sustainability focus: {facts['n_sustainability']}
- FSIs with educational components: {facts['n_education']}
- FSIs organising events or workshops: {facts['n_events']}""",
        "types",
    )
    print("    ✓ FSI types")

    # ── Operational and funding ───────────────────────────────────────────────
    n_noncommercial = total - facts['n_commercial']
    prose["operational"] = checked_write(
        graph_answers.get("operational_levels", {}).get("answer", ""),
        f"""- Total FSIs: {total}
- Government-funded: {facts['n_govt']}
- Council-supported: {facts['n_council']}
- NGO-led: {facts['n_ngo']}
- Community-led (grassroots): {facts['n_community']}
- Commercial or social enterprise: {facts['n_commercial']}
- Non-commercial (voluntary/charity): {n_noncommercial} — note: this is total minus commercial
- Mention crowdfunding or donations: {facts['n_crowdfund']}
- Volunteer-based: {facts['n_volunteer']}
- Cooperative model: {facts['n_coop']}
- Formally registered (NGO/charity): {facts['n_ngo_formal']}
- Informally run by community groups: {facts['n_informal']}
- Affiliated with known food networks: {facts['n_network']}""",
        "operational",
    )
    print("    ✓ operational levels")

    # ── Reach and activity ────────────────────────────────────────────────────
    audiences = "; ".join(
        f"{a}: {n}" for a, n in facts["audience_counts"].items() if n > 0
    )
    prose["reach"] = checked_write(
        graph_answers.get("popularity", {}).get("answer", ""),
        f"""- High activity FSIs: {facts['pop_counts'].get('high', 0)}
- Medium activity FSIs: {facts['pop_counts'].get('medium', 0)}
- Low activity FSIs: {facts['pop_counts'].get('low', 0)}
- Total quantified volume referenced (meals/kg/members): {facts['total_volume']} units
- FSIs collaborating with farms/restaurants/stores: {facts['n_collab']}
- Target audiences mentioned: {audiences}""",
        "reach",
    )
    print("    ✓ reach and activity")

    # ── Digital presence and accessibility ───────────────────────────────────
    _lang_names = {
        "es": "Spanish/Catalan", "ca": "Catalan", "it": "Italian",
        "fr": "French", "de": "German", "pt": "Portuguese",
    }
    _lang_note = ""
    if language != "en":
        _lang_label = _lang_names.get(language, language.upper())
        _lang_note = (
            f"\n- Source content language: {_lang_label} (non-English); all FSI websites"
            f" were processed cross-lingually and summarised in English"
            f"\n- Multilingual coverage: cross-lingual LLM processing enabled"
            f" English-language summarisation of {_lang_label}-language source material"
        )

    prose["digital"] = checked_write(
        "",
        f"""- All {total} FSIs have a digital presence (website scraped)
- FSIs with online/digital coordination mentioned: {facts['n_online']}
- FSIs operating from physical locations: {facts['n_physical']}
- FSIs listing operating hours or frequency: {facts['n_hours_listed']}{_lang_note}
- Data gap: accessibility features not consistently present on FSI websites""",
        "digital",
    )
    print("    ✓ digital presence")

    # ── Notable initiatives ───────────────────────────────────────────────────
    prose["notable"] = checked_write(
        graph_answers.get("notable_initiatives", {}).get("answer", ""),
        f"""- Total FSIs in dataset: {total}
- Most active districts: {top}
- High-popularity FSIs: {facts['pop_counts'].get('high', 0)}
- FSIs with educational components: {facts['n_education']}
- FSIs organising events: {facts['n_events']}""",
        "notable",
    )
    print("    ✓ notable initiatives")

    print("  ✓ All sections synthesized")
    return prose