import contextlib

import variables as var
import data_scrapping as d_sp

with open(f"{var.path_to_logs}/{var.scrapping_log_file}", "w") as o:
    with contextlib.redirect_stdout(o):
        data_previews = d_sp.parse_pages(2, var.guardian_max_page)

print(len(data_previews))

d_sp.save_preview_data(data_previews, f"{var.previews_all_file}")
