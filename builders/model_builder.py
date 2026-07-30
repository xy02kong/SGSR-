from model.SGSR import SGSR


def build_model(
    model_name,
    num_classes,
    **kwargs,
):
    if model_name.lower() == "sgsr":
        return SGSR(
            num_classes=num_classes,
            **kwargs,
        )

    raise NotImplementedError(
        f"Unsupported model: {model_name}"
    )
