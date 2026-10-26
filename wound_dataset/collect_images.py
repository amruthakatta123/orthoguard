"""
collect_images.py
Downloads candidate wound images from Bing for 5 sports surgeries x 3 classes.

Install:   pip install icrawler
Run:       python collect_images.py

Output:    raw/<class>/<surgery>/*.jpg

IMPORTANT: search results are noisy. The folder a file lands in is only a GUESS
based on the query. You must manually review every image (see README.md) and
move mislabeled or irrelevant ones before training.
"""
import os
from icrawler.builtin import BingImageCrawler

MAX_PER_QUERY = 60  # keep modest; quality drops fast past ~60

# surgery -> list of body-site / incision terms
SURGERIES = {
    "acl": ["ACL reconstruction", "knee ACL surgery incision"],
    "rotator_cuff": ["rotator cuff repair", "shoulder arthroscopy"],
    "meniscus": ["meniscus repair", "knee arthroscopy"],
    "labrum": ["shoulder labrum repair", "shoulder stabilization surgery"],
    "achilles": ["Achilles tendon repair", "Achilles tendon surgery"],
}

# class -> phrases appended to the surgery term
CLASSES = {
    "normal": [
        "incision healing day 7",
        "incision healing 2 weeks sutures",
        "healed surgical scar",
        "stitches removed incision",
    ],
    "concerning": [
        "incision redness swelling",
        "incision mild infection",
        "wound irritation around stitches",
        "stitch abscess",
    ],
    "urgent": [
        "surgical site infection",
        "wound dehiscence",
        "infected incision pus",
        "wound necrosis after surgery",
    ],
}


def main():
    for cls, phrases in CLASSES.items():
        for surgery, terms in SURGERIES.items():
            out_dir = os.path.join("raw", cls, surgery)
            os.makedirs(out_dir, exist_ok=True)
            file_idx = len(os.listdir(out_dir))  # resume-friendly
            for term in terms:
                for phrase in phrases:
                    query = f"{term} {phrase}"
                    print(f"[{cls}/{surgery}] {query}")
                    crawler = BingImageCrawler(
                        storage={"root_dir": out_dir},
                        feeder_threads=1,
                        parser_threads=2,
                        downloader_threads=4,
                    )
                    crawler.crawl(
                        keyword=query,
                        max_num=MAX_PER_QUERY,
                        file_idx_offset=file_idx,
                    )
                    file_idx = len(os.listdir(out_dir))
                    # log queries so you can describe your sourcing in your writeup
                    with open("queries_log.txt", "a") as f:
                        f.write(f"{cls}\t{surgery}\t{query}\n")


if __name__ == "__main__":
    main()
