import argilla as rg
from datasets import Dataset


rg.init(api_url="https://burtenshaw-argilla-latest.hf.space", api_key="admin.apikey")


def pull_argilla_dataset(
    dataset_name: str, workspace: str = "admin"
) -> "RemoteFeedbackDataset":
    return rg.FeedbackDataset.from_argilla(name=dataset_name, workspace=workspace)
