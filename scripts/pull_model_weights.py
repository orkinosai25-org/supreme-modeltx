"""
pull_model_weights.py — Download model weights from Azure Blob Storage
                         using a Managed Identity (no credentials on disk).

Usage:
    python scripts/pull_model_weights.py \
        --storage-account smtxstorage \
        --container smtxmodels \
        --blob-prefix t101/ \
        --dest-dir /mnt/model

Requires: pip install azure-storage-blob azure-identity
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smtx.pull_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull SMTX model weights from Azure Blob.")
    parser.add_argument("--storage-account", required=True, help="Azure Storage account name.")
    parser.add_argument("--container", default="smtxmodels", help="Blob container name.")
    parser.add_argument("--blob-prefix", default="t101/", help="Blob path prefix for model files.")
    parser.add_argument("--dest-dir", default="/mnt/model", help="Local destination directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobServiceClient

    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    account_url = f"https://{args.storage_account}.blob.core.windows.net"
    logger.info("Authenticating via Managed Identity to %s …", account_url)
    credential = ManagedIdentityCredential()
    svc = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = svc.get_container_client(args.container)

    blobs = list(container_client.list_blobs(name_starts_with=args.blob_prefix))
    if not blobs:
        logger.warning("No blobs found with prefix '%s' in container '%s'.", args.blob_prefix, args.container)
        return

    logger.info("Found %d blob(s) — downloading to %s …", len(blobs), dest)
    for blob in blobs:
        relative_path = blob.name[len(args.blob_prefix):]
        local_path = dest / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size == blob.size:
            logger.info("  SKIP (up-to-date): %s", blob.name)
            continue

        logger.info("  Downloading: %s → %s", blob.name, local_path)
        with open(local_path, "wb") as f:
            data = container_client.download_blob(blob.name).readall()
            f.write(data)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
