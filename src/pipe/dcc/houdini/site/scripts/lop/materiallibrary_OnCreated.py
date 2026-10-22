import hou

try:
    me: hou.Node = kwargs["node"]  # type: ignore[name-defined] # noqa: F821
    tabmenumask = me.parm("tabmenumask")
    assert tabmenumask is not None
    # "moonray" is the render mask of the DW_MOONRAY shader VOPs
    tabmenumask.set("risnet moonray USD  ^hmtlx* MaterialX collect parameter subnet")
except Exception:  # in case this is created as a locked node
    pass
