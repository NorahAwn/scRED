"""
Download the Wamsley et al. 2024 PsychENCODE ASD single-cell atlas from Synapse
for scRED second-cohort validation.

Project: syn51032009  (Wamsley ASD single-cell atlas)

Before running:
  1. Revoke the old token you pasted, create a new PAT:
     https://www.synapse.org/#!PersonalAccessTokens:
  2. Set it as an environment variable (do NOT hard-code it):
       Windows PowerShell:  setx SYNAPSE_AUTH_TOKEN "your_new_token"
       (then open a NEW terminal)
  3. pip install --upgrade synapseclient
"""

import os
import synapseclient
from synapseclient import Synapse
import synapseutils

# --- Configuration ---
PROJECT_ID = "syn51032009"                       # Wamsley ASD atlas (root)
PROJECT_DIR = r"F:\Norah\GeneGCN"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "wamsley_2024")
os.makedirs(DATA_DIR, exist_ok=True)

# Read token from environment (never store it in the file)
AUTH_TOKEN = os.environ.get("SYNAPSE_AUTH_TOKEN")


def login():
    if not AUTH_TOKEN:
        raise SystemExit(
            "SYNAPSE_AUTH_TOKEN not set.\n"
            "Run:  setx SYNAPSE_AUTH_TOKEN \"your_token\"  then open a new terminal."
        )
    syn = Synapse()
    syn.login(authToken=AUTH_TOKEN)
    print("Logged into Synapse as:", syn.getUserProfile()["userName"])
    return syn


def list_contents(syn):
    """Print every file in the project with its Synapse ID, so you can see
    exactly what will be downloaded before pulling GBs of data."""
    print(f"\nContents of {PROJECT_ID}:\n" + "-" * 60)
    files = []
    for dirpath, subfolders, filenames in synapseutils.walk(syn, PROJECT_ID):
        folder_name, folder_id = dirpath
        for fname, fid in filenames:
            print(f"{fid}\t{folder_name}/{fname}")
            files.append((fid, fname))
    print("-" * 60)
    print(f"{len(files)} files found.\n")
    return files


def download_all(syn):
    """Recursively download the whole project, preserving folder structure.
    Resumes partial downloads; skips files already present and unchanged."""
    print(f"Downloading {PROJECT_ID} -> {DATA_DIR}")
    syn.get(
        PROJECT_ID,
        downloadLocation=DATA_DIR,
        downloadFile=True,
        ifcollision="keep.local",   # don't re-download files you already have
        followLink=True,
    )
    print("Download complete.")


def download_selected(syn, syn_ids):
    """Download only specific files by Synapse ID (use IDs from list_contents)."""
    for sid in syn_ids:
        print(f"Downloading {sid} ...")
        syn.get(sid, downloadLocation=DATA_DIR, ifcollision="keep.local")
    print("Selected downloads complete.")


if __name__ == "__main__":
    syn = login()

    # Step 1: always list first so you know the real file IDs
    files = list_contents(syn)

    # Step 2: choose ONE of the following.

    # (a) Download everything in the project:
    download_all(syn)

    # (b) OR comment out (a) above and download only what you need, e.g.:
    # download_selected(syn, ["syn_REAL_ID_1", "syn_REAL_ID_2"])