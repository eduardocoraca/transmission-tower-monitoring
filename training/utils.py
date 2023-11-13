def get_filename(model: str, cable: int, **model_args) -> str:
    args_str = ""
    for k, v in model_args.items():
        args_str += f"_{k}{v}"
    return f"{model}_cable{cable}{args_str}"
