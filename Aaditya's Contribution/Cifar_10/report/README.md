# LaTeX Report

Main source:

- `repvit_cifar10_report.tex`

The report references figures from `../visuals/`, so keep the current repository structure when compiling.

## Build

If LaTeX is installed, compile from the `report/` directory:

```bash
pdflatex repvit_cifar10_report.tex
pdflatex repvit_cifar10_report.tex
```

Or with `latexmk`:

```bash
latexmk -pdf repvit_cifar10_report.tex
```

## Included Content

- motivation for reducing the initial stride on CIFAR-10
- rationale for keeping SE blocks early and removing them later for latency
- experiment setup
- quantitative comparison table
- training, trade-off, architecture, and pipeline visuals
