# Multimodal Output-Capability Audit — Full vs. baseline_v2

Practical output-capability comparison of the existing, already-rendered HTML reports for both conditions, across all five cities. **This is not a test of factual accuracy and is not evidence for GraphRAG's causal contribution to report quality** -- it records only what each stored artefact contains and states about itself.

## City-by-city evidence

### barcelona

- **Full**: 4 representative image(s); map present=True (alt: 'FSI locations in Barcelona'); coordinate map=False; administrative choropleth=True; district chart present=True; district basis=administrative_boundary_geometry; images with direct source link=4/4; images with literal URL displayed=4/4.

- **baseline_v2**: 0 representative image(s); map present=True (alt: 'Coordinate map of retained FSIs in Barcelona'); coordinate map=True; administrative choropleth=False; district chart present=True; district basis=page_stated_or_coordinate_derived (caption: 'Retained FSI counts by Barcelona district, using page-stated neighbourhoods/addresses and the submitted coordinates where needed.'); images with direct source link=0/0; images with literal URL displayed=0/0.

### brighton

- **Full**: 4 representative image(s); map present=True (alt: 'FSI locations in Brighton & Hove'); coordinate map=False; administrative choropleth=True; district chart present=True; district basis=administrative_boundary_geometry; images with direct source link=4/4; images with literal URL displayed=4/4.

- **baseline_v2**: 4 representative image(s); map present=True (alt: 'Coordinate map of retained Brighton and Hove FSIs'); coordinate map=True; administrative choropleth=False; district chart present=True; district basis=page_stated_or_coordinate_derived (caption: 'District bars use broad coordinate-derived zones because the input supplies no administrative boundaries. Nine retained rows without a complete coordinate pair are excluded from these bars.'); images with direct source link=4/4; images with literal URL displayed=0/4.

### dublin

- **Full**: 4 representative image(s); map present=True (alt: 'FSI locations in Dublin'); coordinate map=False; administrative choropleth=True; district chart present=True; district basis=administrative_boundary_geometry; images with direct source link=4/4; images with literal URL displayed=4/4.

- **baseline_v2**: 4 representative image(s); map present=True (alt: 'Coordinate map of retained Dublin FSIs'); coordinate map=True; administrative choropleth=False; district chart present=True; district basis=page_stated_or_coordinate_derived (caption: 'Retained FSI count by postal district. Page-stated districts were preferred; neighbourhood and supplied-coordinate inference was used where a page did not state a district.'); images with direct source link=4/4; images with literal URL displayed=0/4.

### london

- **Full**: 4 representative image(s); map present=True (alt: 'FSI locations in London'); coordinate map=False; administrative choropleth=True; district chart present=True; district basis=administrative_boundary_geometry; images with direct source link=4/4; images with literal URL displayed=4/4.

- **baseline_v2**: 4 representative image(s); map present=True (alt: 'Coordinate map of retained London food sharing initiatives'); coordinate map=True; administrative choropleth=False; district chart present=True; district basis=page_stated_or_coordinate_derived (caption: 'District or borough counts are derived from the supplied coordinates and page-stated locations for the 27 retained initiatives assignable to one local district. Six citywide or multi-borough initiatives are excluded from this bar chart rather than forced into one district.'); images with direct source link=0/4; images with literal URL displayed=0/4.

### milan

- **Full**: 4 representative image(s); map present=True (alt: 'FSI locations in Milan'); coordinate map=False; administrative choropleth=True; district chart present=True; district basis=administrative_boundary_geometry; images with direct source link=4/4; images with literal URL displayed=4/4.

- **baseline_v2**: 4 representative image(s); map present=True (alt: 'Coordinate map of retained Milan FSIs'); coordinate map=True; administrative choropleth=False; district chart present=True; district basis=page_stated_or_coordinate_derived (caption: 'Counts by inferred neighbourhood/district grouping derived from page-stated locations and submitted coordinates; these are analytical groupings rather than official administrative boundaries.'); images with direct source link=0/4; images with literal URL displayed=0/4.

## Pooled counts

- Map present: Full 5/5, baseline_v2 5/5.
- Administrative-boundary choropleth present: Full 5/5, baseline_v2 0/5.
- Coordinate/point map present: Full 0/5, baseline_v2 5/5.
- District chart present: Full 5/5, baseline_v2 5/5.
- Total representative images: Full 20, baseline_v2 16 (baseline_v2: 0 in barcelona).
- Images with a direct source hyperlink: Full 20/20, baseline_v2 8/16.
- Images with the literal URL displayed as visible text (not hidden behind generic anchor text): Full 20/20, baseline_v2 0/16.

## Distinctions this audit deliberately preserves

- **"Map exists" vs. "administrative choropleth exists"**: every Full report has a map, and every one of those maps is an administrative-boundary choropleth (district polygons coloured by density) -- but the map's own alt text/caption say 'FSI locations', which reads as a point map despite the underlying rendering never plotting individual coordinates (code-traced, not inferred). Every baseline_v2 report also has a map, and every one is a genuine coordinate/point map (per its own stated alt text and caption) -- baseline_v2 has zero administrative-choropleth maps, by its own account, since it has no district boundary data available to it.
- **"Images exist" vs. "image provenance is explicitly traceable"**: representative images exist in 4/5 baseline_v2 reports (0 in barcelona) and all 5 Full reports. Direct source hyperlinks exist in Full (100% of its images) and in brighton/dublin's baseline_v2 images, but NOT in london's (no `<a>` at all) or milan's (plain-text attribution, no link) baseline_v2 images.
- **Clickable generic links vs. literal URLs displayed to the reader**: Full's template prints the raw URL itself as the visible link text for every image that has one. Every baseline_v2 image link found (brighton, dublin) uses generic anchor text ('Source page'/'source page') with the real URL hidden in the href attribute, never shown to the reader as text.

## What this does NOT establish

- This audit does not test or claim that either condition's reported figures, district assignments, or captions are factually correct.
- Neither condition is described as more accurate here -- accuracy was not independently tested in this audit.
- This is a practical output-capability comparison only, and is NOT evidence for the causal contribution of GraphRAG to report quality -- Full's use of administrative-boundary geometry and baseline_v2's use of raw coordinates reflects a difference in available input data (a districts.geojson file vs. none), not necessarily a difference attributable to GraphRAG itself.
