import os
import json

from typing import Any

import numpy as np
from openai import OpenAI
from pandas import read_csv
from pandas import DataFrame
from operator import itemgetter
from scipy.spatial import distance

# Pseudocode:
# 1 - Create a Semantic Search class
# 2 - Define "fetch into json" instance method
# 2 - Define "create_texts" static method for this particular case
# 3 - Define "create embeddings" instance method
# 4 - Define "find closest n" static method

emb_client = OpenAI()
class SemanticSearch:
    def __init__(self, client):
        self.client = client
        self.dataframe = None

    def __call__(self, data_file:str, query:str):
        self.dataframe = read_csv(data_file)

        data = self._fetch_into_json()

        data_texts: list[str] = [self.create_data_texts(data_dict=d) for d in data]
        data_embeddings: list[str] = self.create_embedding(texts=data_texts)[0]

        query_embeddings = self.create_embedding(texts=query)[0]

    def _fetch_into_json(self) -> list[dict[str, Any]]:
        json_data = self.dataframe.to_json(orient='records')
        data:list[dict[str, Any]] = json.loads(json_data)
        return data

    @staticmethod
    def create_data_texts(data_dict: dict[str, Any]) -> str:
        (title,
         short_description,
         features) = itemgetter(
            'title',
            'short_description',
            'features')(data_dict)
        return f"""Title: {title}
        Description: {short_description}
        Features: {', '.join(features)}"""


    def create_embedding(self, texts: list | str)->list[Any]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small"
            ,inputs=texts,
        )
        response_json = response.model_dump()
        return [response_data["embedding"] for response_data in response_json["data"]]


    @staticmethod
    def find_closest_n(query_vector: np.ndarray, embeddings: np.ndarray, n:int)->list[dict[str, Any]]:
        distances: list[dict[str, Any]] = []
        for index, embedding in enumerate(embeddings):
            dist = distance.cosine(query_vector, embedding)
            distances.append({"index": index, "distance": dist})
        sorted_distances = sorted(distances, key=lambda d: d["distance"])
        return sorted_distances[:n]