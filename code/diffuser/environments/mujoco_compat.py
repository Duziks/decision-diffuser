def get_mujoco_data(env):
    """Return MuJoCo data for legacy mujoco-py and modern Gym APIs."""
    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim, "data"):
        return sim.data

    data = getattr(env, "data", None)
    if data is not None:
        return data

    raise AttributeError("MuJoCo environment exposes neither sim.data nor data")
