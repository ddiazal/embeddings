import os
import json

from typing import Any
from openai import OpenAI
from pandas import read_csv
from pandas import DataFrame
from scipy.spatial import distance

# Pseudocode:
# 1 - Create a Semantic Search class
# 2 - Define "create_texts" static method for this particular case
# 3 - Define "create embeddings" instance method
# 4 - Define "find closest n" static method

emb_client = OpenAI()
class SemanticSearch:
    def __init__(self, client):
        self.client = client
        self.dataframe = None