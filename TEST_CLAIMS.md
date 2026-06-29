# Recommended Test Claims (Mendelian Mini Corpus)

These 12 claims are designed to work with the bundled **Mendelian disease mini
corpus** (`semantic_scholar/manifests/mendelian_v1.json`). The corpus was curated
and then expanded (~5,400 papers with abstracts) specifically so that the papers
the pipeline retrieves for these claims are available locally. Use them to smoke-
test the full claim-processing pipeline without a full Semantic Scholar release.

## Quick copy/paste

Paste into the **Run Claims** box on the home page (one claim per line), or upload
as a `.txt` file:

```
Biallelic pathogenic variants in PAH cause phenylalanine hydroxylase deficiency, and early dietary phenylalanine restriction can prevent severe neurocognitive outcomes in classic phenylketonuria.
Pathogenic CFTR variants cause cystic fibrosis, but pulmonary disease severity can vary among individuals with the same CFTR genotype because of modifier genes and environmental factors.
FBN1-related Marfan syndrome is inherited in an autosomal dominant manner, and aortic root aneurysm or dissection is a central clinical risk.
In Huntington disease, HTT CAG repeat length is associated with age at onset, but genetic modifiers in DNA maintenance pathways also influence onset.
Out-of-frame DMD variants generally cause Duchenne muscular dystrophy, whereas in-frame DMD variants are more often associated with the milder Becker muscular dystrophy phenotype.
Sickle cell disease is caused by at least one HbS HBB allele plus a second pathogenic HBB variant, and fetal hemoglobin modifiers can substantially alter disease severity.
Biallelic pathogenic variants in HEXA cause Tay-Sachs disease through deficient beta-hexosaminidase A activity and GM2 ganglioside accumulation.
Most familial hypercholesterolemia is caused by autosomal dominant pathogenic variants in LDLR, APOB, or PCSK9, while LDLRAP1-associated familial hypercholesterolemia is autosomal recessive.
GBA1 pathogenic variants cause Gaucher disease, and enzyme replacement therapy improves many visceral and hematologic manifestations but is limited for neuronopathic disease because it does not adequately treat central nervous system involvement.
Achondroplasia is caused by activating FGFR3 pathogenic variants, and homozygosity for the common achondroplasia-causing FGFR3 variant is typically lethal.
Germline pathogenic variants in BRCA1 or BRCA2 are inherited in an autosomal dominant manner and increase breast and ovarian cancer risk, but penetrance varies by gene, variant, and family context.
Biallelic ATP7B pathogenic variants cause Wilson disease by impairing copper transport; heterozygous carrier status alone is not sufficient to diagnose Wilson disease.
```

## The claims

| # | Disease | Gene(s) | Inheritance | What it exercises |
|---|---------|---------|-------------|-------------------|
| 1 | Phenylketonuria | PAH | AR | Clear support claim: mechanism + newborn screening + treatment |
| 2 | Cystic fibrosis | CFTR | AR | Causal claim with genotype–phenotype / modifier-gene nuance |
| 3 | Marfan syndrome | FBN1 | AD | Classic gene–disease causality with a key clinical outcome |
| 4 | Huntington disease | HTT | AD | Repeat-expansion mechanism + modifier evidence |
| 5 | Duchenne/Becker MD | DMD | X-linked | Genotype–phenotype "rule" with real exceptions (avoid overclaiming) |
| 6 | Sickle cell disease | HBB, BCL11A | AR / compound het | Mendelian causality + modifier-locus literature |
| 7 | Tay-Sachs disease | HEXA | AR | Classic lysosomal-storage enzyme mechanism |
| 8 | Familial hypercholesterolemia | LDLR, APOB, PCSK9, LDLRAP1 | Mixed AD/AR | Inheritance nuance (penalize "AD only" answers) |
| 9 | Gaucher disease | GBA1 | AR | Treatment effect **with** a mechanistic limitation |
| 10 | Achondroplasia | FGFR3 | AD (homozygous lethal) | Gain-of-function with zygosity nuance |
| 11 | Hereditary breast/ovarian cancer | BRCA1, BRCA2 | AD predisposition | Penetrance / risk-interpretation nuance |
| 12 | Wilson disease | ATP7B | AR | Includes a negative control (carrier ≠ affected) |

## Maintaining / re-expanding the corpus

Helper scripts under `scripts/` (each takes the live S2 API; a
`SEMANTIC_SCHOLAR_API_KEY` is read from settings):

| Script | Purpose |
|--------|---------|
| `expand_mini_corpus.py` | Harvest papers-with-abstracts from the live API and rewrite the extracts + manifest. |
| `verify_mini_corpus_access.py` | Confirm retrieved papers resolve to local content (mirrors the pipeline's accessibility check). |

The mini release is built from the manifest by the downloader itself:
`python -m semantic_scholar.utils.downloader --mini` (and `--mini --verify` checks
the local release against the manifest) — the same path the Data page's New
Release wizard uses.

Typical refresh: `expand_mini_corpus.py --per-disease 450` →
`python -m semantic_scholar.utils.downloader --mini` → `verify_mini_corpus_access.py`.
Then re-run a claim (e.g. the BRCA one above) from the UI to confirm end to end.
