from uuid import uuid4

import argilla as rg
from datasets import Dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask
from jsonschema import validate

from src import utils


def generate_and_push_dpo_dataset(dataset):
    """Generate a dataset for DPO based on the Pydantic dataset. This time we want
        to train a model to generate JSON schemas based on use cases.
        Args:
            dataset (Dataset): The Pydantic dataset.
        Returns:
            RemoteFeedbackDataset: The remote dataset for DPO on Argilla.
    """
    generator_system_prompt = (
        "You an expert JSON schema developer, specialising in JSON schema."
        "You are given a use case for a specific application entity."
        "You write only JSON schemas and do not introduce the code with prose."
        "Define an entity in JSON that conforms to the following schema, based on the use case."
    )
    pipeline = Pipeline(
        generator=OpenAILLM(
            model="gpt-4",
            task=TextGenerationTask(system_prompt=generator_system_prompt),
            prompt_format="openai",
            max_new_tokens=1024,
            num_threads=1,
            temperature=0.0,
        )
    )
    generated_dataset = pipeline.generate(
        dataset=dataset, num_generations=5, batch_size=5, display_progress_bar=True
    )

    generated_dataset = generated_dataset.map(validate_response_json_schema)

    feedback_dataset = rg.FeedbackDataset.for_direct_preference_optimization(
        number_of_responses=2,
        context=False,
        use_markdown=True,
        guidelines=None,
        metadata_properties=None,
        vectors_settings=None,
    )

    records = [_build_record(sample) for sample in generated_dataset]
    feedback_dataset.add_records(records)
    remote_dataset = feedback_dataset.push_to_argilla(
        name=f"json-response-dpo-{uuid4()}", workspace="admin"
    )
    return remote_dataset


def pull_convert_to_datasets(feedback_dataset: "RemoteFeedbackDataset") -> Dataset:
    """Pulls the feedback dataset from Argilla and converts it to a datasets dataset.
    Args:
        feedback_dataset (RemoteFeedbackDataset): The feedback dataset from Argilla.
    Returns:
        Dataset: The dataset of feedback.
    """
    feedback_dataset = feedback_dataset.filter_by(response_status="submitted")
    feedback_dataset = feedback_dataset.pull()

    def convert(record):
        sample = {
            "instruction": record.fields["instruction"],
            "code": record.fields["code"],
            "json_schema": record.fields["json_schema"],
            "generation": record.fields["generation"],
        }
        responses = {
            key: value.value for key, value in record.responses[0].values.items()
        }
        sample.update(responses)
        use_case_prefix = "Define a pydantic class called UseCase for this use case:"
        sample["instruction"] = sample["instruction"].replace(use_case_prefix, "")
        sample["code"] = sample["code"].replace("```python\n", "").replace("\n```", "")
        return sample

    dataset = Dataset.from_list(list(map(convert, feedback_dataset)))
    return dataset


def execute_generated_pydantic_models(sample):
    """Executes the generated Pydantic models and validates the JSON schema.
    Args:
        sample (dict[str, Any]): A generated Pydantic model.
    Returns:
        dict[str, Any]: The sample with the JSON schema and validation.
    """
    sample = dict(sample)
    try:
        exec(sample["code"])
        json_schema = str(eval(sample["usecase_class"]).schema())
        validity = 1
    except Exception as e:
        json_schema = str(e)
        validity = 0
    instruction = sample["instruction"]
    sample["input"] = f"{instruction}\nOutput Structure:\n{json_schema}"
    sample["json_schema"] = json_schema
    sample["valid_json_schema"] = validity
    return sample


def validate_response_json_schema(sample) -> dict[str, dict[str, bool]]:
    """Validates the generated JSON schema."""
    json_schema = sample["input"].split("\nOutput Structure:\n")[1]
    response1 = sample["generations"][0]
    response2 = sample["generations"][1]

    def valid_json_schema(json_object):
        try:
            validate(instance=eval(json_object), schema=eval(json_schema))
            return True
        except Exception as e:
            return False

    return {
        "schema": {
            "response1": valid_json_schema(response1),
            "response2": valid_json_schema(response2),
        }
    }


def _build_record(sample) -> rg.FeedbackRecord:
    """Builds a feedback record with JSON rendering"""
    prompt = sample["input"]
    response1 = _render_json_in_markdown(sample["generations"][0])
    response2 = _render_json_in_markdown(sample["generations"][1])
    json_schema = _render_json_in_markdown(sample["json_schema"])
    suggestions = []
    for response_name, validity in sample["schema"].items():
        if validity:
            suggestions.append({"rank": 1, "value": response_name})
            break
    return rg.FeedbackRecord(
        fields={
            "prompt": prompt,
            "json_schema": json_schema,
            "response1": response1,
            "response2": response2,
        },
        suggestions=[
            rg.SuggestionSchema(question_name="preference", value=suggestions)
        ],
        metadata=sample["schema"],
    )


def _render_json_in_markdown(json_string: str) -> str:
    return f"```json\n{json_string}\n```"
