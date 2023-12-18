from re import A
import time
from argparse import ArgumentParser
from datetime import datetime

from dotenv import load_dotenv

from src.prompts import generate_prompts
from src.pydantic_models import generate_pydantic_models, push_pydantic_to_argilla
from src.json_schema import (
    generate_and_push_dpo_dataset,
    pull_convert_to_datasets,
    execute_generated_pydantic_models,
)

from src import utils

load_dotenv()


def main(
    number_of_use_cases: int = 5,
    max_wait_seconds: int = 3600,
    existing_pydantic_dataset_name: str | None = None,
):

    # Generate Pydantic dataset based on use cases
    generated_use_cases = generate_prompts()
    generated_pydantic_dataset = generate_pydantic_models(use_cases=generated_use_cases)
    remote_pydantic_feedback_dataset = push_pydantic_to_argilla(
        pipeline_dataset=generated_pydantic_dataset,
        argilla_dataset_name=existing_pydantic_dataset_name,
    )

    # Wait for Human Feedback on Pydantic dataset
    submitted_records = 0
    start_time = datetime.now()
    while submitted_records <= number_of_use_cases:
        print(f"Waiting for {number_of_use_cases} human responses...")
        submitted = remote_pydantic_feedback_dataset.filter_by(
            response_status="submitted"
        )
        submitted_records = len(submitted.records)
        # Don't wait too long 
        if (datetime.now() - start_time).seconds > max_wait_seconds:
            return
        time.sleep(20)

    # Pull Pydantic dataset from Argilla
    pydantic_feedback_dataset = utils.pull_argilla_dataset(
        dataset_name=remote_pydantic_feedback_dataset.name
    )
    pydantic_dataset = pull_convert_to_datasets(pydantic_feedback_dataset)

    # Validate Pydantic code and generate DPO dataset for feedback
    pydantic_dataset = pydantic_dataset.map(execute_generated_pydantic_models)
    pydantic_dataset = pydantic_dataset.filter(
        lambda sample: sample["valid_json_schema"]
    )
    remote_dpo_feedback_dataset = generate_and_push_dpo_dataset(pydantic_dataset)

    print(remote_dpo_feedback_dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--number_of_use_cases",
        type=int,
        default=5,
        help="Number of use cases to generate",
    )
    parser.add_argument(
        "--max_wait_seconds",
        type=int,
        default=3600,
        help="Maximum number of seconds to wait for human feedback",
    )
    parser.add_argument(
        "--existing_pydantic_dataset_name",
        type=str,
        default=None,
        help="Name of existing Pydantic dataset on Argilla",
    )
    args = parser.parse_args()
    main(
        number_of_use_cases=args.number_of_use_cases,
        max_wait_seconds=args.max_wait_seconds,
        existing_pydantic_dataset_name=args.existing_pydantic_dataset_name,
    )
