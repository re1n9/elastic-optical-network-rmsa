# Define Modulation Formats as the project assignment
MODULATION_FORMATS = {
    "DP-16QAM": {"rate":400, "slots":6,"max_length":500, "cost":3.7},
    "SC-DP-16QAM": {"rate":200, "slots":3,"max_length":700, "cost":2.0},
    "SC-DP-QPSK": {"rate":100, "slots":3,"max_length":2000, "cost":1.5}
}

#choose modulation function
def choose_modulation(path_length, demand_gbps):
    best_mod = None
    best_slots = float("inf")
    best_cost = float("inf")

    for mod, props in MODULATION_FORMATS.items():
        if path_length <= props["max_length"]:
            rate = props["rate"]
            num_lightpaths = -(-demand_gbps // rate)
            total_slots = int(props["slots"] * num_lightpaths)
            total_cost = int(props["cost"] * num_lightpaths)

            if best_mod is None or rate > MODULATION_FORMATS[best_mod]["rate"]:
                best_mod, best_slots, best_cost = mod, total_slots, total_cost
    return best_mod, best_slots, best_cost if best_mod else (None, None, None)
