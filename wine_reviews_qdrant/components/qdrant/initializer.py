from typing import Text

import lightning as L
import json

from zipfile import ZipFile

import kaggle
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.grpc import Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Batch
from sentence_transformers import SentenceTransformer


class QdrantWineReviewsDataInitializer(L.LightningWork):
    """
    Performs data initialization including vectorization and pushing the data
    into Qdrant server collection.
    """

    def __init__(
        self,
        qdrant_host: Text,
        qdrant_collection: Text,
        model_name: Text,
        http_port: int = 6333,
        grpc_port: int = 6334,
    ):
        super().__init__(parallel=True)
        self.qdrant_host = qdrant_host
        self.qdrant_collection = qdrant_collection
        self.model_name = model_name
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.is_initialized = False

    def run(self, *args, **kwargs):
        if self.is_initialized:
            return

        self.is_initialized = True
        qdrant_client = QdrantClient(
            self.qdrant_host, port=self.http_port, grpc_port=self.grpc_port
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

        # Determine the vector dimensionality
        model = SentenceTransformer(self.model_name)
        print("Loaded selected sentence transformer", self.model_name)
        dimensionality = model.get_sentence_embedding_dimension()

        # Check whether the collection is already created, and if so, then just
        # skip the initialization, because it has been already made before.
        try:
            qdrant_client.get_collection(self.qdrant_collection)
            print("Collection already exists, so it won't be created")
            return
        except UnexpectedResponse:
            print("Creating Qdrant collection")
            qdrant_client.recreate_collection(
                collection_name=self.qdrant_collection,
                vector_size=dimensionality,
                distance=Distance.Cosine,
            )

        # Put all the vectors with the corresponding attributes into Qdrant by
        # dividing the data frame into chunks
        for start_index in range(0, wine_reviews_df.shape[0], 1000):
            print("Upserting vectors from ", start_index)
            chunk_df = wine_reviews_df.iloc[start_index : start_index + 1000]
            embeddings = model.encode(chunk_df["description"].tolist()).tolist()
            payloads = json.loads(chunk_df.to_json(orient="records"))
            qdrant_client.upsert(
                self.qdrant_collection,
                points=Batch(
                    ids=chunk_df.index.tolist(), vectors=embeddings, payloads=payloads
                ),
            )
