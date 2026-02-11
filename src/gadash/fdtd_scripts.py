from __future__ import annotations


def gds_import_script(
    *,
    gds_path: str,
    cell_name_primary: str,
    cell_name_fallback: str,
    layer_map: str = "1:0",
    target_material: str = "Si3N4 (Silicon Nitride) - Phillip",
    z_min: float = 0.0,
    z_max: float = 600e-9,
    preclean_names: list[str] | None = None,
) -> str:
    # Keep script syntax as close as possible to the original notebook flow.
    # NOTE: Lumerical script parser is picky; we intentionally emit direct
    #       string literals (double quotes) instead of variables.
    gds_path_escaped = gds_path.replace("\\", "/")
    pre = ""
    for nm in (preclean_names or []):
        s = str(nm).strip()
        if not s:
            continue
        s = s.replace('"', '\\"')
        pre += f'try{{ select("{s}"); delete; }} catch(errPre);\n'
    return f"""
try{{ select("IMPORTED_GDS"); delete; }} catch(errMsg);
{pre}

import_ok = 0;
used_cell = "";
import_err = "";

try{{
    gdsimport("{gds_path_escaped}", "{cell_name_primary}", "{layer_map}", "{target_material}", {float(z_min)}, {float(z_max)});
    set("name", "IMPORTED_GDS");
    import_ok = 1;
    used_cell = "{cell_name_primary}";
}} catch(import_err);

if (import_ok == 0) {{
    import_err = "";
    try{{
        gdsimport("{gds_path_escaped}", "{cell_name_fallback}", "{layer_map}", "{target_material}", {float(z_min)}, {float(z_max)});
        set("name", "IMPORTED_GDS");
        import_ok = 1;
        used_cell = "{cell_name_fallback}";
    }} catch(import_err);
}}
"""


def extract_spectra_script(
    *,
    trans_1: str = "Trans_1",
    trans_2: str = "Trans_2",
    trans_3: str = "Trans_3",
) -> str:
    # Keep same extraction convention as original notebook (three monitors).
    return (
        f"m1 = '{trans_1}';\n"
        f"m2 = '{trans_2}';\n"
        f"m3 = '{trans_3}';\n"
        "if (haveresult(m1)) { T1=transmission(m1); f_vec=getdata(m1,'f'); } else { T1=0; f_vec=0; }\n"
        "if (haveresult(m2)) { T2=transmission(m2); } else { T2=0; }\n"
        "if (haveresult(m3)) { T3=transmission(m3); } else { T3=0; }\n"
    )
