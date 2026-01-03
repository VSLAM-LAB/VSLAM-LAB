
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

@staticmethod
def _get_dataset_issue(issue_id: str, *, dataset_name: str, size_gb: int = 0, website: str = "", yaml_file: str = "") -> dict:
    factories = {
        "complete_dataset": lambda: {
            "name": "Complete dataset",
            "description": (
                 f"The '{dataset_name}' dataset does not permit downloading individual sequences."
            ),
            "solution": f"Download the full dataset ({size_gb} GB).",
            "mode": f"{GREEN}automatic download{RESET}",
        },
        "license_required": lambda: {
            "name": "License acceptance required",
            "description": (
                f"Access to '{dataset_name}' requires prior acceptance of the license terms."
            ),
            "solution": "Open the dataset page, accept the license, then retry the download.",
            "mode": f"{YELLOW}user action required{RESET}",
        },
        "api_token": lambda: {
            "name": "Api token required",
            "description": (
                f"Access to '{dataset_name}' requires use of an API TOKEN."
            ),
            "solution": f"Register at '{website}' to get an API TOKEN, then set it in '{yaml_file}'.",
            "mode": f"{YELLOW}user action required{RESET}",
        },
            
    }

    try:
        return factories[issue_id]()  # build on demand so fields can use args
    except KeyError:
        available = ", ".join(sorted(factories.keys()))
        raise ValueError(f"Unknown issue_id '{issue_id}'. Available: {available}") from None
