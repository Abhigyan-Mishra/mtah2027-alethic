# Upstream Development Provenance Notice

This standalone repository contains the complete, inspectable competition submission package for the **More Than A Hack 2027** competition (Phase 1: Ball Possession).

### Notice Regarding Internal References
In `TECHNICAL_REPORT.md`, citations referencing internal documentation, review rounds (e.g. *Round 61*), ledger notes (e.g. *Note N-M4.1*), internal engine modules (e.g. *frames.py*, *kloppy_parity.json*), or development commit SHAs (e.g. *fec63f5*, *05d4c14*, *0fcf8bc*, *ee5864c*, *5f8589b*, *11cba74*, *5404f54*) refer to the private upstream engineering and audit repository (`Sovereign-Engine`).

The complete, self-contained public competition pipeline is defined and certified by the 26 files listed in `SUBMISSION_MANIFEST.sha256`. All models, weights, and submission CSVs can be verified, trained, and executed directly using only the files in this repository.
