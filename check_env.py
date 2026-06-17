<<<<<<< HEAD
"""Environment check for NCERT + PYQ semantic search demo."""

from __future__ import annotations

import importlib

REQUIRED_PACKAGES = {
    "fitz": "PyMuPDF",
    "sentence_transformers": "sentence-transformers",
    "chromadb": "chromadb",
    "streamlit": "streamlit",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "sklearn": "scikit-learn",
}


def main() -> int:
    missing = []

    print("Checking Python environment dependencies...\n")
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] {package_name}")
        except Exception:
            print(f"[MISSING] {package_name}")
            missing.append(package_name)

    print()
    if missing:
        print("Missing packages detected:")
        for pkg in missing:
            print(f"- {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return 1

    print("All required packages are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
=======
"""Environment check for NCERT + PYQ semantic search demo."""

from __future__ import annotations

import importlib

REQUIRED_PACKAGES = {
    "fitz": "PyMuPDF",
    "sentence_transformers": "sentence-transformers",
    "chromadb": "chromadb",
    "streamlit": "streamlit",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "sklearn": "scikit-learn",
}


def main() -> int:
    missing = []

    print("Checking Python environment dependencies...\n")
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] {package_name}")
        except Exception:
            print(f"[MISSING] {package_name}")
            missing.append(package_name)

    print()
    if missing:
        print("Missing packages detected:")
        for pkg in missing:
            print(f"- {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return 1

    print("All required packages are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
>>>>>>> b87913a (Initial commit)
