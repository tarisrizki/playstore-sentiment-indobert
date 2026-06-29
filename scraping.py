# %% [markdown]
# # Scraping Ulasan Google Play Store

# %%
# !pip install google-play-scraper polars
import time
import random
import logging
from datetime import datetime
import polars as pl
from google_play_scraper import reviews, Sort

APPS = {
    "com.tokopedia.tkpd": "Tokopedia",
    "com.gojek.app": "Gojek",
    "id.dana": "DANA",
}

BATCH_SIZE = 500
TARGET_PER_APP = 6_000
MAX_BATCHES = TARGET_PER_APP // BATCH_SIZE + 5
OUTPUT_FILE = "reviews.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# %%
def scrape_app_reviews(app_id: str, app_name: str, target: int) -> list[dict]:
    all_reviews: list[dict] = []
    continuation_token = None
    log.info(f"▶ Mulai scraping: {app_name} ({app_id}) — target {target}")

    for batch_num in range(1, MAX_BATCHES + 1):
        try:
            result, continuation_token = reviews(
                app_id, lang="id", country="id", sort=Sort.NEWEST,
                count=BATCH_SIZE, continuation_token=continuation_token,
            )
        except Exception as e:
            log.warning(f"  ⚠ Error batch {batch_num}: {e}")
            time.sleep(random.uniform(5, 10))
            continue

        if not result: break

        for r in result:
            all_reviews.append({
                "content": r.get("content", ""),
                "score": r.get("score", 0),
                "at": r.get("at", ""),
                "userName": r.get("userName", ""),
                "thumbsUpCount": r.get("thumbsUpCount", 0),
                "reviewCreatedVersion": r.get("reviewCreatedVersion", ""),
                "app_id": app_id,
                "app_name": app_name,
            })

        log.info(f"  📦 Batch {batch_num:>3d} — diperoleh {len(result):>4d} │ total: {len(all_reviews):>5d}")
        if len(all_reviews) >= target or continuation_token is None: break
        time.sleep(random.uniform(1.5, 3.5))

    return all_reviews[:target]

def label_sentiment(score: int) -> str:
    if score <= 2: return "Negatif"
    if score == 3: return "Netral"
    return "Positif"

# %%
start_time = datetime.now()
all_data: list[dict] = []

for app_id, app_name in APPS.items():
    app_reviews = scrape_app_reviews(app_id, app_name, TARGET_PER_APP)
    all_data.extend(app_reviews)

# %%
df = pl.DataFrame(all_data)
df = df.with_columns(pl.col("score").map_elements(label_sentiment, return_dtype=pl.Utf8).alias("sentiment"))
df = df.filter(pl.col("content").is_not_null() & (pl.col("content").str.len_chars() > 0))
df = df.unique(subset=["content"])
df.write_csv(OUTPUT_FILE)

log.info(f"Total review unik : {df.height:,}")
log.info(f"Waktu eksekusi    : {datetime.now() - start_time}")

# %%
display(df.head(5))
