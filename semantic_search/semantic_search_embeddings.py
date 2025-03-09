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
SECRET_KEY: str = os.getenv("OPI_OPENAI_API_KEY")
emb_client = OpenAI(
    api_key=SECRET_KEY,
)
class SemanticSearch:
    def __init__(self, client = emb_client):
        self.__client = client
        self.dataframe = None

    def __call__(self, data_file:str, query:str, file_type: str = "csv"):
        if file_type == "csv":
            self.dataframe: DataFrame = read_csv(data_file)
            data = self._fetch_into_json()
        else:
            with open(data_file) as f:
                data = json.load(f)

        data_texts: list[str] = [self.create_data_texts(data_dict=d) for d in data]
        data_embeddings: list[float] = self.create_embedding(texts=data_texts)[0]

        query_embeddings: list[float] = self.create_embedding(texts=query)[0]

        hits = self.find_closest_n(query_embeddings, data_embeddings, n=3)
        sem_search: list = []
        for hit in hits:
            sem_search.append(data[hit["index"]])

        return sem_search

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
        response = self.__client.embeddings.create(
            model="text-embedding-3-small"
            ,inputs=texts,
        )
        response_json = response.model_dump()
        return [response_data["embedding"] for response_data in response_json["data"]]


    @staticmethod
    def find_closest_n(query_vector: list[float], embeddings: list[float], n:int)->list[dict[str, Any]]:
        distances: list[dict[str, Any]] = []
        for index, embedding in enumerate(embeddings):
            dist = distance.cosine(query_vector, embedding)
            distances.append({"index": index, "distance": dist})
        sorted_distances = sorted(distances, key=lambda d: d["distance"])
        return sorted_distances[:n]