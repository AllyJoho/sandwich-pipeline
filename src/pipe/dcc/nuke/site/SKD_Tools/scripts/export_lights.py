import os
import json

import nuke


def _default_output_path():
    script_path = nuke.root().name()
    if script_path == "Root":
        return ""
    return os.path.join(os.path.dirname(script_path), "light_edits.json")


def run():
    output_path = nuke.getFilename(
        "Export light grades to",
        "*.json",
        default=_default_output_path(),
        type="save",
    )
    if not output_path:
        return

    data = {}

    for node in nuke.root().nodes():
        if node.Class() == "Grade":
            name = node.name()
            mult_val = node["multiply"].value()

            # Case A: It's a single float (all RGB are the same)
            if isinstance(mult_val, float):
                rgb = [mult_val, mult_val, mult_val]

            # Case B: It's a list/tuple (user split the channels)
            else:
                # Slicing [:3] takes index 0, 1, and 2
                rgb = list(mult_val[:3])

            light_name = name.replace("_grade", "")
            data[light_name] = {"multiply": rgb}

    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        nuke.message(f"Exported to:\n{output_path}")
    except Exception as e:
        nuke.message(f"Write failed: {str(e)}")
