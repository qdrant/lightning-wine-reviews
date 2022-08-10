import logging
from functools import partial

import gradio as gr
import lightning as L

from typing import Text, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class WineReviewsUI(L.LightningWork):
    """
    Serve model predictions with Gradio UI.
    """

    inputs = [
        gr.components.Textbox(label="Wine description query", lines=2),
        gr.components.Slider(7, 120, value=75, label="Maximum price"),
    ]
    outputs = [
        gr.components.Label(label="Title"),
        gr.components.Label(label="Variety"),
        gr.components.Textbox(label="Description", lines=3),
        gr.components.Number(label="Price"),
        gr.components.Number(label="Similarity"),
    ]
    examples = [
        ["This is a fairly complex taste", 120],
        ["Cheap, yet drinkable", 10],
        ["Pineapple", 120],
        ["Pineapple, but way cheaper", 25],
    ]

    def __init__(
        self,
        model_name: Text,
        qdrant_host: Text,
        qdrant_collecton: Text,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.qdrant_host = qdrant_host
        self.qdrant_collection = qdrant_collecton
        self._qdrant_client = None
        self._model = None

    def predict(
        self, query: Text, max_price: float
    ) -> Tuple[Text, Text, Text, float, float]:
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(host=self.qdrant_host)
        embedding = self._model.encode(query)
        results = self._qdrant_client.search(
            collection_name=self.qdrant_collection,
            query_vector=embedding,
            query_filter=Filter(
                must=[FieldCondition(key="price", range=Range(lte=max_price))]
            ),
            limit=1,
        )
        if len(results) == 0:
            return None, None, None, None, None
        top_entry = results[0]
        return (
            top_entry.payload["title"],
            top_entry.payload["variety"],
            top_entry.payload["description"],
            top_entry.payload["price"],
            top_entry.score,
        )

    def build_model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def run(self, *args, **kwargs):
        if self._model is None:
            self._model = self.build_model()
        fn = partial(self.predict, *args, **kwargs)
        fn.__name__ = self.predict.__name__
        interface = gr.Interface(
            fn=fn,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            allow_flagging="never",
            title="Find the next bottle",
            description="Provide a textual query on the left side and see the "
            "most similar wine in the database",
        )
        interface.launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=False,
        )
