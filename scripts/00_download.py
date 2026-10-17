"""Download CESNET-TLS-Year22 through cesnet-datazoo (resumable).

    python scripts/00_download.py --data-root data --size XS

Sizes: XS=10M, S=25M, M=50M, L=100M flows sampled over the year. Start with XS.
The file size is printed before the download starts, so you can abort (Ctrl+C) if disk space is short;
re-running resumes a partial download.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--size", default="XS", choices=["XS", "S", "M", "L"])
    a = ap.parse_args()

    from cesnet_datazoo.datasets import CESNET_TLS_Year22
    ds = CESNET_TLS_Year22(a.data_root, size=a.size, database_checks_at_init=False)
    print("\nDownloaded to :", ds.database_path)
    print("Services      :", len(ds.available_classes))
    print("Days available:", len(ds.available_dates))
    print("Periods       :", [p for p in ds.time_periods if p.startswith(("W-", "M-"))][:8], "...")
    print("\nNext: python scripts/01_explore.py --data-root", a.data_root, "--size", a.size)


if __name__ == "__main__":
    main()
