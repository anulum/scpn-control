# Build

From this directory, using PATH-resolved pdfTeX and BibTeX:

```bash
export SOURCE_DATE_EPOCH=1787386155
export FORCE_SOURCE_DATE=1
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

The expected review artefact is `manuscript.pdf`. Run the commands in a
disposable copy during clean-tree verification; routine TeX auxiliaries are not
repository artefacts.
