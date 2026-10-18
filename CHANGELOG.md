# Changelog

## v0 — 2026-10

First public release.

- PolyDETR v0 weights (23.2 M parameters, safetensors) trained on about 110,000 synthetic pages.
- Local web app: PDF and image upload (multi-page PDF and TIFF, HEIC), page-by-page results, four extraction modes
  (polygon, filled rectangle, straightened, perspective), minimum score slider (default 0.40), PNG and ZIP downloads.
- PDF rendering with PDFium (pypdfium2); everything runs on the CPU and stays on the computer.
