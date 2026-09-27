import os

file_path = 'functions/main.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

firebase_export = """
# ============================================================
# FIREBASE FUNCTIONS EXPORT
# ============================================================
from firebase_functions import https_fn, options

api = https_fn.on_request(
    app,
    memory=options.MemoryOption.GB_4,
    timeout_sec=300,
    max_instances=1
)
"""

if "FIREBASE FUNCTIONS EXPORT" not in content:
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(firebase_export)
