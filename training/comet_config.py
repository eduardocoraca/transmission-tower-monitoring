import os

from dotenv import load_dotenv

load_dotenv()


def get_comet_config(project_env: str) -> dict[str, str | None]:
    return {
        "api_key": os.getenv("COMET_API_KEY"),
        "workspace": os.getenv("COMET_WORKSPACE"),
        "project_name": os.getenv(project_env),
        "model_name": os.getenv("COMET_MODEL_NAME"),
        "model_version": os.getenv("COMET_MODEL_VERSION"),
    }
