# Evidence hierarchy

`reference.json` is the frozen Stage-A source/property ledger (78 records),
including the **retired** Belimo reference. It supplies material proxies to the
new assembly. It is not the active runtime configuration or final gap status.

`config/model.json` is the active cassette configuration. `active-parameters.csv`
maps its scalar settings to evidence/status and epistemic/design ranges.
Intervals are not confidence intervals or supplier tolerances. Layer properties
and intervals remain in the frozen ledger; bridge geometry/contact assumptions
are explicit in model.json. `additional-sources.json` and
`new-source-provenance.json` contain the new primary sources/data hashes.

Run `python run.py audit`: the authoritative current gap status is
`results/cassette-v2/evidence-audit.json`. All five requested final-device topics
still have explicit evidence/validation gaps despite the implemented code.
