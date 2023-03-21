import ujson as json

from lib.dataclasses import Sample, SignalDCT, SignalSpectrum, SignalTime, SingleSample


def get_sample_from_json(path: str, sample_id: str, json_mappings: dict) -> Sample:
    with open(path) as f:
        json_dict = json.load(f)

    sample_dict = {"metadata": {}}
    for key in json_dict.keys():
        if key in json_mappings.keys():
            field = json_mappings[key]
            sample_dict[field] = json_dict[key]
        else:
            sample_dict["metadata"][key] = json_dict[key]

    sample_per_cable = {}
    for cable in [1, 2, 3, 4]:
        cable_sample_per_channel = {}
        for channel in ["x", "y", "z"]:
            cable_sample_per_channel[channel] = SignalDCT(
                coefficients=sample_dict[f"cable{cable}_{channel}"],
                indexes=sample_dict[f"index_cable{cable}_{channel}"],
                original_length=1024 * 16,
                ts=sample_dict["metadata"]["dt"],
            )
        sample_per_cable[f"c{cable}"] = SingleSample(
            signal_x_dct=cable_sample_per_channel["x"],
            signal_y_dct=cable_sample_per_channel["y"],
            signal_z_dct=cable_sample_per_channel["z"],
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
