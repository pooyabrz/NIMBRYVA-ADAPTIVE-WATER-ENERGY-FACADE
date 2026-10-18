# Data attribution and scope

`haechler-fig1c-raw.csv`, `haechler-fig4b.csv`, `haechler-fig4c.csv` are extracted
numerical sheets from **Iwan Haechler (2021), Extended Data for “Exploiting
Radiative Cooling for Uninterrupted 24-hour Water Harvesting from the
Atmosphere”, ETH Zurich**, DOI https://doi.org/10.3929/ethz-b-000469811 .

The repository explicitly licenses this dataset under **Creative Commons
Attribution-NonCommercial 4.0 International**:
https://creativecommons.org/licenses/by-nc/4.0/ . Preserve attribution, indicate
extraction/QC changes and obtain appropriate permission for uses beyond the
license. The project's code license does not relicense these data.

Extraction changes: percentages converted to fractions for Figure 1C; raw
values retained, including 13 values above 1; physical/trusted-range flags
added. Figure 4B and 4C preserve the original numerical flux/time columns.
Values are source experiments, not performance of the proposed cassette.

`ASTMG173.csv` is the reference solar spectrum from the U.S. national-laboratory
resource at https://www.nlr.gov/grid/solar-resource/spectra-am1.5 . It describes
one specified reference atmosphere/orientation. It is not a measured time
series for these cities. Consult the source data disclaimer when reusing it.

Original download URLs, SHA256 and extraction scope are recorded in
`config/evidence/new-source-provenance.json`. Commercial product PDFs and full
articles are linked rather than redistributed.
