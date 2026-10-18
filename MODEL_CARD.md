# Model card: PolyDETR v0

## Summary

| | |
|---|---|
| Task | Detect the images contained in a document page (PDF render or phone photo) and return one polygon per image |
| Classes | one: *image* (photos, charts, plots, diagrams, drawings, maps, floor plans, screenshots, figures with their labels and callouts) |
| Architecture | DETR-style detector: ResNet-18 backbone, hybrid encoder, 6-layer decoder, 40 queries, each predicting a score and a closed contour of 32 points; denoising queries during training |
| Parameters | 23.2 M (93 MB in safetensors) |
| Input | the page letterboxed to 1024 × 768 |
| Output | score and polygon per detection; NMS at IoU 0.7, polygon simplified (1 px tolerance) |
| Speed | about 0.5 s per page on a recent laptop CPU (Apple M-series, measured in the app) |
| Weights | exponential moving average of the training weights, step 56,648 |
| License | CC BY-SA 4.0 (weights); Apache-2.0 (code) |

## Training data

Only synthetic data. About **110,000 generated pages** (books, scientific papers, magazines, slides, reports,
dashboards, galleries, catalogues; white, tinted, dark and patterned backgrounds; one to three columns) containing
about **283,000 images**:

- 61% are figures generated on the fly and therefore all different: bar/line/pie charts, curves, schematic diagrams,
  floor plans, technical drawings, screenshots, figures with callouts (labels and leader lines drawn directly on the
  page), groups of images linked by arrows;
- the rest are drawn, cropped and transformed from **19,444 openly licensed images**: Open Images (CC BY 2.0),
  Wikimedia Commons (public domain, CC0, CC BY, CC BY-SA), Unsplash (Unsplash License). Per-asset sources, authors and
  licenses are in [attribution/training-assets.csv](attribution/training-assets.csv).

Target polygons follow fixed rules so that they are learnable: photos are outlined exactly; complex figures (charts,
diagrams, drawings) get a constant 2 pt margin and include their titles, legends, labels and callouts; captions
outside the figure are excluded; a target never covers body text or another image; cut-out shapes in the middle of
text get a free contour of up to 24 vertices.

At training time each page is randomly turned into a photo: perspective and rotation, lighting and shadows, paper and
desk textures, double-page spreads, fingers, pens and phones covering parts of the page, photocopy and JPEG artefacts.

Training: AdamW (lr 1e-4, backbone 1e-5), batch 12, warmup-stable-decay schedule with a final 1-sqrt decay over
10,000 steps, EMA 0.999, about 680,000 page views, on one RTX 4090 for about 11 hours.

## Evaluation

On 1,500 held-out synthetic pages (3,884 images, threshold 0.5): 98% of the images found with IoU ≥ 0.5 and 97% with
IoU ≥ 0.75; 0.025 false positives per page; median contour distance 4.0 px and 90th percentile 12.9 px at 300 dpi
(the previous internal model: 9.6 and 30.4 px on the same pages).

**There is no annotated real-world benchmark yet.** On real pages inspected by eye the model is clearly better than
general layout detectors on phone photos of pages, and comparable on clean digital PDFs; its confidence is lower than
on synthetic data, which is why the app uses a default threshold of 0.40.

## Known limitations

- Infographics and composite figures (pictograms, mini charts, text boxes and arrows) are often split into pieces.
- Arrow-shaped bullets and decorative arrows near a figure can be absorbed into it; adjacent images can be merged.
- Simple line drawings embedded in dense pages can get low scores.
- Large blank areas are occasionally detected as images.
- Not designed for tables, formulas or text blocks.

## License of the weights

The training pages are synthesised from openly licensed images, 12.6% of which are under CC BY-SA licenses and a
handful (29 of 19,444, 0.1%) under GPL, GFDL or Free Art licenses. Whether trained weights are a derivative work of
their training images is an open legal question; to respect the share-alike terms in any case, the weights are
released under **CC BY-SA 4.0**, with attribution to the image sources in [ATTRIBUTION.md](ATTRIBUTION.md). Future
versions will exclude share-alike and copyleft sources from the generator.
