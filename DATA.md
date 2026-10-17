# Dataset: CESNET-TLS-Year22

The data is **not stored in this repository** (too large). It is listed in `.gitignore` under `data/`.

## Easiest: automatic download (recommended)

```
python scripts/00_download.py --data-root data --size XS
```
Every team member runs this once on their own machine. The download resumes if interrupted.

| Size | Flows | Use for |
|---|---|---|
| XS | 10 M | development, all experiments in `run_all` |
| S | 25 M | final runs if time/hardware allow |
| M / L | 50 M / 100 M | not needed |

## Manual download (if the script fails)

Files for size XS, from the CESNET DataZoo server:

* Database: https://liberouter.org/datazoo/download?bucket=cesnet-tls-year22&file=CESNET-TLS-Year22-XS.h5
* Service map: https://liberouter.org/datazoo/download?bucket=cesnet-tls-year22&file=servicemap.csv

Put both files in `data/XS/` so the layout is:

```
data/XS/CESNET-TLS-Year22-XS.h5
data/XS/servicemap.csv
```

## Original sources

* Zenodo (full dataset record): https://zenodo.org/records/10608607
* Dataset page: https://netmon.fit.cvut.cz/datasets/cesnet-tls-year22/
* Tool used to read it (`cesnet-datazoo`): https://github.com/CESNET/cesnet-datazoo
* Paper: https://doi.org/10.1038/s41597-024-03927-4

## Dataset facts

* TLS flows from a 100 Gbps ISP backbone, all of 2022, 180 services in 24 categories.
* Per flow: first 30 packets (inter-packet time, direction, size), flow statistics, and some TLS fields. No payload.
* Week 10 (7-13 Mar 2022): the flow exporter was updated and the packet-sequence distribution changed. See README section 5.
* Please cite the paper above when using the data.
