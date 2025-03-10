import os
import json

from typing import Any

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
        print(f'Search results for "{query}"')
        for hit in hits:
            product = data[hit["index"]]
            sem_search.append(product["title"])

        return sem_search

    def _fetch_into_json(self) -> list[dict[str, Any]]:
        json_data = self.dataframe.to_json(orient='records')
        data:list[dict[str, Any]] = json.loads(json_data)
        return data

    @staticmethod
    def create_data_texts(data_dict: dict[str, Any]) -> str:
        """

        :param data_dict:
        :return:
        """
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
        """

        :param texts:
        :return:
        """
        response = self.__client.embeddings.create(
            model="text-embedding-3-small"
            ,input=texts
        )
        response_json = response.model_dump()
        return [response_data["embedding"] for response_data in response_json["data"]]


    @staticmethod
    def find_closest_n(query_vector: list[float], embeddings: list[float], n:int)->list[dict[str, Any]]:
        """

        :param query_vector:
        :param embeddings:
        :param n:
        :return:
        """
        distances: list[dict[str, Any]] = []
        for index, embedding in enumerate(embeddings):
            # Calculate the cosine distance between the query vector and embedding
            dist = distance.cosine(query_vector, embedding)
            # Append the distance and index to distances
            distances.append({"index": index, "distance": dist})
        # Sort distances by the distance key
        sorted_distances = sorted(distances, key=lambda d: d["distance"])
        # Return the first n elements in distances_sorted
        return sorted_distances[:n]