"""Thin Code Ocean entry point for ophys ROI classification.

All logic lives in the ``aind-ophys-classifier-library`` package; this
wrapper only parses settings (CLI / environment) and invokes ``run``.
"""

from aind_ophys_classifier_library.job import run

if __name__ == "__main__":
    run()
