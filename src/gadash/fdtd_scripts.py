from __future__ import annotations


def gds_import_script(
    *,
    gds_path: str,
    cell_name: str,
    layer_map: str = "1:0",
) -> str:
    # LumAPI scripts are executed via fdtd.eval(script).
    gds_path_escaped = gds_path.replace("\\", "/")
    return (
        f"gds_path = '{gds_path_escaped}';\n"
        f"cell_name = '{cell_name}';\n"
        f"layer_map = '{layer_map}';\n"
        "switchtolayout;\n"
        "gdsimport(gds_path, cell_name, layer_map);\n"
    )


def extract_spectra_script() -> str:
    # Template-dependent extraction.
    #
    # Convention for this project:
    # - A single monitor object named 'monitor' (update if needed)
    # - getdata(monitor,'f') -> frequency or wavelength vector
    # - getdata(monitor,'T') -> transmission array for 4 outputs
    #   expected shape either (4,N) or (N,4)
    #
    # The python bridge will reshape T into RGGB: (2,2,N) using order [0,1;2,3].
    return (
        "monitor_name = 'monitor';\n"
        "f_vec = getdata(monitor_name,'f');\n"
        "T = getdata(monitor_name,'T');\n"
    )

