"""Run the full pipeline:  python code/pipeline.py [fetch|extract|build|all]"""
import sys

import build_map_data
import extract
import fetch_news

STEPS = {"fetch": fetch_news.main, "extract": extract.main, "build": build_map_data.main}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STEPS if step == "all" else [step]):
        STEPS[name]()
