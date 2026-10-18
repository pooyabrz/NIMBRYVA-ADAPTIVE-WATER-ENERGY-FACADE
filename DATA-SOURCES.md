# Data provenance and attribution

Meteorological and shortwave forcing: ERA5 reanalysis, produced by ECMWF for the Copernicus Climate Change Service, obtained through Open-Meteo's Historical Weather API with an explicit ERA5 selection. Credit: Copernicus Climate Change Service / ECMWF and Open-Meteo.

Longwave forcing: NASA Prediction Of Worldwide Energy Resources (POWER) Project hourly `ALLSKY_SFC_LW_DWN`, Renewable Energy community, UTC. Credit: NASA POWER Project at NASA Langley Research Center.

Exact request URLs, retrieval timestamps, byte counts and SHA256 digests accompany each unmodified source response in `data/raw/*.provenance.json`. Source response headers and original units are retained in JSON. Normalization metadata identifies the different data footprints, time-label assumption and lack of facade resolution.

The project's code license does not relicense the underlying third-party data. The source providers' data and service terms apply. Original terms and citation guidance:

- https://open-meteo.com/en/terms
- https://open-meteo.com/en/docs/historical-weather-api
- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- https://power.larc.nasa.gov/docs/acknowledgments/

The initial dataset is one realized reference year, 2024, not a long-term climate climatology or forecast.

New material/optical datasets and separate license/attribution:
`data/materials/NOTICE.md`; extraction and source hashes:
`config/evidence/new-source-provenance.json`. The ETH material data are
CC BY-NC 4.0; the code license does not replace that license.
