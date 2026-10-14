import os

import nuke

_THIRD_PARTY = os.environ["DCC_NUKE_THIRD_PARTY"]

nuke.pluginAddPath(
    os.path.join(
        _THIRD_PARTY, "NukeSurvivalToolkit_publicRelease", "NukeSurvivalToolkit"
    )
)

nuke.pluginAddPath(os.path.join(_THIRD_PARTY, "scripts"))
nuke.pluginAddPath("./SKD_Tools")

# Comps made before UHD keep this format, saved in their script.
nuke.addFormat("1920 1080 Bobo_aspect_ratio")

# Renders stay 1080p until finals; each Read's Reformat sizes them up. Nuke's
# default proxy, scale 0.5, shows a UHD comp at 1080p.
nuke.knobDefault("Root.format", "UHD_4K")

# color management
nuke.knobDefault("Root.colorManagement", "OCIO")

# spektrafilm reads ARRI LogC4 and writes a display image unless told otherwise.
# Comp is ACEScg, and OpenDRT is applied by the viewer and the MOV write, so each
# node reads ACEScg and hands its film render back in ACEScg (RCM/ACES role).
_SPEKTRAFILM_NODES = (
    "OFXorg.spektrafilm_v0",
    "OFXorg.spektrafilm.flow_v0",
    "OFXorg.spektrafilm.diffuse_v0",
    "OFXorg.spektrafilm.lens_v0",
)
for _node in _SPEKTRAFILM_NODES:
    for _knob in ("inputColorSpace", "quickInputColorSpace", "rcmInputColorSpace"):
        nuke.knobDefault(f"{_node}.{_knob}", "ACEScg")
    nuke.knobDefault(f"{_node}.outputRole", "RCM/ACES (Beta)")

# ntsc-rs filters the signal it is given; analog video carries a gamma-encoded
# signal, not linear light.
nuke.knobDefault("OFXwtf.vala:NtscRs_v1.SrgbGammaCorrect", "true")
