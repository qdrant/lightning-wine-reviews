import json
from pathlib import Path
from typing import Dict, Text
from zipfile import ZipFile

import kaggle
import pandas as pd
import lightning as L
from qdrant_client import QdrantClient
from qdrant_client.grpc import Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Batch
from sentence_transformers import SentenceTransformer

from wine_reviews_qdrant import QdrantServerComponent, WineReviewsUI


class QuaterionFineTuningApp(L.LightningFlow):
    """
    A demo application showing how to fine tune a transformer-based encoder
    to include user preferences.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    QDRANT_HOST = "localhost"
    QDRANT_COLLECTION = "wine-reviews"
    VECTOR_DIMENSIONALITY = 384

    def __init__(self) -> None:
        super().__init__()
        current_dir = Path().resolve()
        self.qdrant_server = QdrantServerComponent(
            "v0.9.0", volume_location=current_dir / "qdrant_storage"
        )
        self.wine_reviews_ui = WineReviewsUI(
            model_name=self.MODEL_NAME,
            qdrant_host=self.QDRANT_HOST,
            qdrant_collecton=self.QDRANT_COLLECTION,
        )
        self.is_initialized = False

    def run(self):
        # Set up Qdrant backend and initialize the database by putting all the
        # vectors into the specified collection. Run the initialization only
        # once, so the collection is not overwritten.
        self.qdrant_server.run()
        if self.qdrant_server.server_running:
            self.initialize_qdrant()

        # Run the UI and allow querying
        self.wine_reviews_ui.run()

    def configure_layout(self) -> Dict[Text, Text]:
        tabs = [
            {"name": "Wine Search", "content": self.wine_reviews_ui.url},
        ]
        return tabs

    def initialize_qdrant(self):
        if self.is_initialized:
            return
        self.is_initialized = True

        qdrant_client = QdrantClient(self.QDRANT_HOST)

        # Check whether the collection is already created, and if so, then just
        # skip the initialization, because it has been already made before.
        try:
            qdrant_client.get_collection(self.QDRANT_COLLECTION)
            print("Collection already exists, so it won't be created")
            return
        except UnexpectedResponse:
            print("Creating Qdrant collection")
            qdrant_client.recreate_collection(
                collection_name=self.QDRANT_COLLECTION,
                vector_size=self.VECTOR_DIMENSIONALITY,
                distance=Distance.Cosine,
            )

        # Download the wine reviews dataset
        print("Downloading the wine-reviews dataset from Kaggle")
        kaggle_api_client = kaggle.KaggleApi()
        kaggle_api_client.authenticate()
        kaggle_api_client.dataset_download_file(
            "zynicide/wine-reviews", "winemag-data-130k-v2.csv"
        )

        # Load the dataset from the downloaded zip file
        print("Loading the wine-reviews dataset")
        with ZipFile("winemag-data-130k-v2.csv.zip") as zip_file:
            source_file = zip_file.open("winemag-data-130k-v2.csv")
            wine_reviews_df = pd.read_csv(source_file, index_col=0)

        # Put all the vectors with the corresponding attributes into Qdrant by
        # dividing the data frame into chunks
        model = SentenceTransformer(self.MODEL_NAME)
        print("Loaded selected sentence transformer", self.MODEL_NAME)
        for start_index in range(0, wine_reviews_df.shape[0], 1000):
            print("Upserting vectors from ", start_index)
            chunk_df = wine_reviews_df.iloc[start_index: start_index + 1000]
            embeddings = model.encode(chunk_df["description"].tolist()).tolist()
            payloads = json.loads(chunk_df.to_json(orient="records"))
            qdrant_client.upsert(
                self.QDRANT_COLLECTION,
                points=Batch(
                    ids=chunk_df.index.tolist(), vectors=embeddings, payloads=payloads
                ),
            )


if "__main__" == __name__:
    app = L.LightningApp(QuaterionFineTuningApp())
