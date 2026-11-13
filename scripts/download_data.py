"""
PaySim Dataset Downloader and Setup Script.
Downloads the official PaySim dataset (6.36M rows) and outputs to
data/PS_20174392719_1491204439457_log.csv.
"""
import os
import sys
import requests
import pyarrow.parquet as pq
import pandas as pd

CSV_PATH = "data/PS_20174392719_1491204439457_log.csv"
PARQUET_PATH = "data/financial_txns.parquet"
DATASET_URL = "https://huggingface.co/datasets/purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection/resolve/main/financial_txns.parquet"

def download_dataset():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 10_000_000:
        print(f"[+] Dataset already exists at {CSV_PATH} ({os.path.getsize(CSV_PATH)} bytes). Skipping download.")
        return

    os.makedirs("data", exist_ok=True)
    if not os.path.exists(PARQUET_PATH) or os.path.getsize(PARQUET_PATH) < 200_000_000:
        print(f"[*] Downloading PaySim parquet from {DATASET_URL}...")
        resp = requests.get(DATASET_URL, stream=True)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 * 5  # 5MB chunks
        with open(PARQUET_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({percent:.1f}%)", end="\r")
        print("\n[+] Download complete.")

    print(f"[*] Converting {PARQUET_PATH} to {CSV_PATH} with memory-efficient batching...")
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    total_rows = parquet_file.metadata.num_rows
    print(f"[*] Total rows in parquet: {total_rows:,}")

    # Inspect column names
    first_batch = next(parquet_file.iter_batches(batch_size=10)).to_pandas()
    print("[*] Parquet columns:", list(first_batch.columns))

    # Convert to CSV in batches
    first = True
    processed = 0
    with open(CSV_PATH, "w") as f:
        for batch in parquet_file.iter_batches(batch_size=500_000):
            df_batch = batch.to_pandas()
            df_batch.to_csv(f, index=False, header=first)
            first = False
            processed += len(df_batch)
            print(f"Wrote {processed:,} / {total_rows:,} rows to CSV...", end="\r")
    print(f"\n[+] Successfully saved {CSV_PATH} ({os.path.getsize(CSV_PATH) // (1024*1024)} MB).")

    # Clean up parquet to save disk space
    if os.path.exists(PARQUET_PATH):
        os.remove(PARQUET_PATH)
        print(f"[+] Cleaned up temporary {PARQUET_PATH}.")

if __name__ == "__main__":
    download_dataset()
