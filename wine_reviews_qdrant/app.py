from pathlib import Path
from typing import Dict, Text

import lightning as L

from wine_reviews_qdrant import (
    QdrantServerComponent,
    QdrantWineReviewsDataInitializer,
    WineReviewsUI,
)


class QuaterionFineTuningApp(L.LightningFlow):
    """
    A demo application showing how to fine tune a transformer-based encoder
    to include user preferences.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    QDRANT_HOST = "localhost"
    QDRANT_COLLECTION = "wine-reviews"

    def __init__(self) -> None:
        super().__init__()
        current_dir = Path().resolve()
        self.qdrant_server = QdrantServerComponent(
            "v0.9.0", volume_location=current_dir / "qdrant_storage"
        )
        self.qdrant_data_initializer = QdrantWineReviewsDataInitializer(
            qdrant_host=self.QDRANT_HOST,
            qdrant_collection=self.QDRANT_COLLECTION,
            model_name=self.MODEL_NAME,
        )
        self.wine_reviews_ui = WineReviewsUI(
            qdrant_host=self.QDRANT_HOST,
            qdrant_collecton=self.QDRANT_COLLECTION,
            model_name=self.MODEL_NAME,
        )
        self.is_initialized = False

    def run(self):
        # Set up Qdrant backend and initialize the database by putting all the
        # vectors into the specified collection. Run the initialization only
        # once, so the collection is not overwritten.
        self.qdrant_server.run()
        if self.qdrant_server.server_running:
            self.qdrant_data_initializer.run()

        # Run the UI and allow querying
        self.wine_reviews_ui.run()

    def configure_layout(self) -> Dict[Text, Text]:
        tabs = [
            {"name": "Wine Search", "content": self.wine_reviews_ui.url},
        ]
        return tabs


if "__main__" == __name__:
    app = L.LightningApp(QuaterionFineTuningApp())
