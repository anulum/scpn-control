# Build

From this directory, using PATH-resolved Pandoc, pdfTeX, and Python with
`pikepdf` installed:

```bash
export SOURCE_DATE_EPOCH="$(jq -r .source_date_epoch submission_metadata.json)"
export FORCE_SOURCE_DATE=1
pandoc manuscript.md \
  --from=markdown \
  --citeproc \
  --bibliography=references.bib \
  --metadata=author:"Miroslav Šotek" \
  --include-in-header=reproducible_pdf.tex \
  --pdf-engine=pdflatex \
  --output=manuscript.raw.pdf
python pdf_metadata.py manuscript.raw.pdf manuscript.pdf
qpdf --check manuscript.pdf
```

The expected review artefact is `manuscript.pdf`. Build in a disposable copy
when verifying a clean tree so no transient TeX files enter the repository.
`pdf_metadata.py` sets the content-based Info and XMP metadata and removes
renderer metadata. `reproducible_pdf.tex` fixes the renderer's trailer ID;
update it from the first 32 hex digits of the manuscript SHA-256, and update
`source_date_epoch` in `submission_metadata.json`, for each revised manuscript.
`papers/verify_submissions.sh` compares the entire PDF byte-for-byte against
the tracked review artefact. Two builds from the same inputs must have the
same SHA-256 digest. Remove the disposable raw PDF and TeX outputs after the
review build.
