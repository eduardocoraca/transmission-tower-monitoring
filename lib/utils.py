import ujson as json
import yaml
from yaml import CLoader as Loader

from lib.dataclasses import (Sample, SignalDCT, SignalSpectrum, SignalTime,
                             SingleSample)

PATH_TO_JSON_MAPPINGS = "/workspaces/project/lib/json_mappings.yml"

with open(PATH_TO_JSON_MAPPINGS, "r") as f:
    JSON_MAPPINGS = yaml.load(f, Loader)


def get_sample_from_json(path: str, sample_id: str, dct: bool = True) -> Sample:
    """Gets a sample from the JSON file and returns a Sample object according
    to the nomenclature in JSON_MAPPINGS.
    """

    with open(path, "r") as f:
        json_dict = json.load(f)

    sample_dict = {"metadata": {}}
    for key in json_dict.keys():
        if key in JSON_MAPPINGS.keys():
            field = JSON_MAPPINGS[key]
            sample_dict[field] = json_dict[key]
        else:
            sample_dict["metadata"][key] = json_dict[key]

    sample_per_cable = {}
    for cable in [1, 2, 3, 4]:
        cable_sample_per_channel = {}
        for channel in ["x", "y", "z"]:
            if dct:
                cable_sample_per_channel[channel] = SignalDCT(
                    coefficients=sample_dict[f"cable{cable}_{channel}"],
                    indexes=sample_dict[f"index_cable{cable}_{channel}"],
                    original_length=1024 * 16,
                    ts=sample_dict["metadata"]["dt"],
                )
            else:
                cable_sample_per_channel[channel] = SignalTime(
                    signal=sample_dict[f"cable{cable}_{channel}"],
                    ts=sample_dict["metadata"]["dt"],
                )
        sample_per_cable[f"c{cable}"] = SingleSample(
            x=cable_sample_per_channel["x"],
            y=cable_sample_per_channel["y"],
            z=cable_sample_per_channel["z"],
            tension=sample_dict[f"cable{cable}_tension"],
        )

    return Sample(
        sample_id=sample_id,
        sample_c1=sample_per_cable["c1"],
        sample_c2=sample_per_cable["c2"],
        sample_c3=sample_per_cable["c3"],
        sample_c4=sample_per_cable["c4"],
        metadata=sample_dict["metadata"],
    )
