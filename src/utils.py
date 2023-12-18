import os

import argilla as rg

API_URL = os.getenv("ARGILLA_API_URL")
API_KEY = os.getenv("ARGILLA_API_KEY")
rg.init(api_url=API_URL, api_key=API_KEY)


def pull_argilla_dataset(
    dataset_name: str, workspace: str = "admin"
) -> "RemoteFeedbackDataset":
    return rg.FeedbackDataset.from_argilla(name=dataset_name, workspace=workspace)
