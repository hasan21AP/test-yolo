from pathlib import Path
import yaml


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "runtime.yaml"
)


# =========================================================
# Load configuration
# =========================================================

def load_config(config_path=None):

    # Use default config if no path is provided
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    else:
        config_path = Path(config_path)

        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path


    # -----------------------------------------------------
    # Check config file
    # -----------------------------------------------------

    if not config_path.exists():

        raise FileNotFoundError(
            f"Config file not found:\n{config_path}"
        )


    # -----------------------------------------------------
    # Read YAML
    # -----------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)


    if config is None:

        raise ValueError(
            "Configuration file is empty."
        )


    # =====================================================
    # Required settings
    # =====================================================

    required_keys = [
        "model",
        "device",
        "imgsz",
        "conf",
        "mode",
        "source_type",
        "source",
    ]


    for key in required_keys:

        if key not in config:

            raise KeyError(
                f"Missing config setting: {key}"
            )


    # =====================================================
    # Detection timing defaults
    # =====================================================

    if "detection_mode" not in config:

        config["detection_mode"] = "continuous"


    if "detection_interval" not in config:

        config["detection_interval"] = 5.0


    # =====================================================
    # Validate detection mode
    # =====================================================

    if config["detection_mode"] not in [
        "continuous",
        "interval"
    ]:

        raise ValueError(
            "detection_mode must be "
            "'continuous' or 'interval'"
        )


    # =====================================================
    # Validate detection interval
    # =====================================================

    config["detection_interval"] = float(
        config["detection_interval"]
    )


    if config["detection_interval"] <= 0:

        raise ValueError(
            "detection_interval must be greater than 0"
        )


    # =====================================================
    # Resolve model path
    # =====================================================

    model_path = Path(
        config["model"]
    )


    if not model_path.is_absolute():

        model_path = (
            PROJECT_ROOT
            / model_path
        )


    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )


    config["model"] = str(
        model_path
    )


    # =====================================================
    # Resolve source
    # =====================================================

    source_type = config["source_type"]


    # -----------------------------------------------------
    # Video file
    # -----------------------------------------------------

    if source_type == "video":

        source_path = Path(
            config["source"]
        )


        if not source_path.is_absolute():

            source_path = (
                PROJECT_ROOT
                / source_path
            )


        if not source_path.exists():

            raise FileNotFoundError(
                f"Video not found:\n{source_path}"
            )


        config["source"] = str(
            source_path
        )


    # -----------------------------------------------------
    # Webcam
    # -----------------------------------------------------

    elif source_type == "webcam":

        config["source"] = int(
            config["source"]
        )


    # -----------------------------------------------------
    # IP / Phone camera
    # -----------------------------------------------------

    elif source_type == "ip_camera":

        config["source"] = str(
            config["source"]
        )


    else:

        raise ValueError(
            f"Unknown source_type: {source_type}"
        )


    # =====================================================
    # Validate mode
    # =====================================================

    if config["mode"] not in [
        "detect",
        "track"
    ]:

        raise ValueError(
            "mode must be 'detect' or 'track'"
        )


    return config