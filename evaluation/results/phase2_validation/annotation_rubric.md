# Phase-2 Classification Validation — Annotation Rubric

Descriptive master's-thesis validation of the Phase-2 `fsi_type` /
`operational_level` labels. You are shown the FULL stored `title` /
`description` / `text` for each record — not the truncated (<=1000
character) excerpt the original classifier actually saw. The model's own
labels are hidden; do not attempt to guess or look them up while annotating.

## STEP 1 — FSI STATUS

`human_fsi_status` must be exactly one of:

- **genuine_fsi** — the stored evidence describes an identifiable
  initiative, programme, organisation, site or recurring activity whose
  purpose includes sharing, redistributing, providing, growing communally,
  swapping, or otherwise making food available through a
  collective/community/service mechanism.
- **not_an_fsi** — the evidence describes something related to food but is
  not itself an initiative of this type, for example: a directory/listing
  page; a policy/document; a news article about initiatives (rather than an
  initiative itself); generic council information; a commercial food
  business with no food-sharing/community-service function; an unrelated
  page.
- **insufficient_evidence** — the stored title/description/text does not
  contain enough information to decide.

## STEP 2 — FSI TYPE

Only if `human_fsi_status = genuine_fsi`. Choose exactly one primary type:

- **food_sharing** — redistributing or sharing surplus/prepared food between
  people or organisations (not a food bank, meal service, or single gifting
  event).
- **food_swapping** — organised exchange of food/produce/seeds between
  participants (swap events, seed swaps).
- **food_gifting** — food made freely available with no exchange expected
  (community fridges, "take what you need" points).
- **community_garden** — communal growing space (allotments, urban gardens,
  orchards) whose primary activity is growing food together.
- **food_bank** — collects and distributes food to people in need, typically
  via a referral or drop-in system.
- **meals_service** — prepares and serves meals directly (soup kitchens,
  community dining, social canteens).
- **food_education** — primary activity is teaching about food (cooking
  classes, nutrition workshops, food-growing training) rather than
  distributing/serving/growing food itself.
- **other** — a genuine FSI that does not fit any of the seven categories
  above. **"other" must NEVER be used for `not_an_fsi` cases** — if the page
  is not an FSI at all, use `human_fsi_status = not_an_fsi` in Step 1
  instead, and leave `human_fsi_type` blank.

If multiple types plausibly apply, select the most central/primary activity
and explain the ambiguity in `human_notes`.

## STEP 3 — OPERATIONAL LEVEL

Only if `human_fsi_status = genuine_fsi`. Choose exactly one:

- **government_funded** — explicit national/state government funding or
  operation.
- **council_supported** — explicit local council/municipal funding, grant,
  or operational support.
- **ngo_led** — run by a registered charity, non-profit, or NGO/foundation.
- **community_led** — grassroots, volunteer-run, or community-group led,
  with no evidence of government/council/NGO/commercial structure.
- **commercial** — social enterprise, CIC, or otherwise operates on a
  trading/membership/fee basis.
- **unknown** — the initiative appears genuine but the operational/funding
  model cannot be determined from the stored evidence.

Use a specific category **only** where the stored evidence directly
supports it. **Do not "best guess" from generic food-related wording alone**
— if the text never mentions funding, ownership, or organisational status,
use `unknown`. If two categories are genuinely plausible, choose the best
supported one and note the ambiguity in `human_notes`.

## STEP 4 — CONFIDENCE

`human_confidence`, exactly one of:

- **high** — explicit evidence directly supports the chosen labels.
- **medium** — reasonable inference from the evidence, but not stated
  explicitly.
- **low** — weak or ambiguous evidence; the label is a plausible best
  reading, not a confident one.

## Notes

- `human_notes` is free text — use it for any ambiguity, multiple plausible
  types/operational levels, or reasoning you want on record.
- This is a descriptive validation sample (75 records total: 50 uniformly
  random "core" records, 25 deliberately edge-case "stress" records that
  must not be treated as representative and must not be pooled with the
  core sample when computing an overall agreement statistic).
