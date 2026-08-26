# Build

From this directory, using PATH-resolved Pandoc and pdfTeX:

```bash
export SOURCE_DATE_EPOCH=1787386155
export FORCE_SOURCE_DATE=1
pandoc manuscript.md \
  --from=markdown \
  --citeproc \
  --bibliography=references.bib \
  --metadata=author:"Miroslav Šotek" \
  --pdf-engine=pdflatex \
  --output=manuscript.pdf
```

The expected review artefact is `manuscript.pdf`. Build in a disposable copy
when verifying a clean tree so no transient TeX files enter the repository.
