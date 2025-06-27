import json

htp_devices = {
    "SM8750": {
        "dsp_arch": "v79",
        "soc_id": 69,
    },
    "SM8650": {
        "dsp_arch": "v75",
        "soc_id": 57,
    },
    "SM8550": {
        "dsp_arch": "v73",
        "soc_id": 43,
    },
    "SC8380": {
        "dsp_arch": "v73",
        "soc_id": 60,
    },
    "SM8475": {
        "dsp_arch": "v69",
        "soc_id": 42,
    },
    "SM8635": {
        "dsp_arch": "v73",
        "soc_id": 68,
    },
    "SM7325": {
        "dsp_arch": "v68",
        "soc_id": 35,
    },
    "SSG2125P": {
        "dsp_arch": "v73",
        "soc_id": 58,
    }
}

def dump_htp_config(soc_name: str, graph_names: list, output_path: str, old_qnn = False, weights_sharing=False):
    if not soc_name in htp_devices.keys():
        raise ValueError(f"Invalid SoC name: {soc_name}")
    if graph_names is None or len(graph_names) == 0:
        raise ValueError("Invalid graph names")
    for i in range(len(graph_names)):
        graph_names[i] = graph_names[i].replace("lib", "").replace("-", "_")

    config = {
        "graphs": {
            "vtcm_mb": 0,
            "O": 3,
            "graph_names": graph_names,
        },
        "devices": [{
            "dsp_arch": htp_devices[soc_name]["dsp_arch"],
            "device_id": 0,
            "soc_id": htp_devices[soc_name]["soc_id"],
            "pd_session": "unsigned",
            "cores": [{
                "core_id": 0,
                "perf_profile": "burst"
            }]
        }],
        "memory": {
            "mem_type": "shared_buffer"
        }
    }

    if soc_name != "SM8635" and soc_name != "SM7325" and soc_name != "SSG2125P":
        config["graphs"]["fp16_relaxed_precision"] = 1

    if not old_qnn:
        config["graphs"] = [config["graphs"]]

    if weights_sharing:
        config["context"] = {"weight_sharing_enabled": True}

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)

def dump_htp_link_config(output_path: str, qnn_sdk_root_path: str):
    link = {
        "backend_extensions":
        {
            "shared_library_path": f"{qnn_sdk_root_path}/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
            "config_file_path": output_path.replace("link.json", "config.json")
        }
    }
    with open(output_path, "w") as f:
        json.dump(link, f, indent=4)
