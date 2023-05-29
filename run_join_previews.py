import contextlib
import pandas as pd

import variables as var
import data_scrapping as d_sp

df_previews = d_sp.download_previews(var.previews_all_file)

for home_team, away_team, preview_date in var.matches_to_swap:
    df_previews = d_sp.swap_teams(df_previews, home_team, away_team, preview_date)

for home_team, away_team, preview_date in var.matches_to_delete:
    df_previews = d_sp.delete_preview(df_previews, home_team, away_team, preview_date)

with open(f"{var.path_to_logs}/{var.joining_log_file}", "w") as o:
    with contextlib.redirect_stdout(o):
        data_info = []
        for season in range(2009, 2022):
            df_data = d_sp.download_season_data(season)
            df_joined = d_sp.join_data_previews(df_data, df_previews)
            df_joined.to_csv(f"{var.path_to_data}/{var.ready_data_name}_{season}.csv", index=False)
            data_info.append([season, len(df_joined['HomeTeam'].dropna()), len(df_joined['home_team'].dropna())])
        df_info = pd.DataFrame(data_info, columns=['Season', 'Matches', 'Previews'])
        df_info.to_csv(f"{var.path_to_data}/{var.season_info_file}")


df_all_seasons = pd.read_csv(f"{var.path_to_data}/{var.ready_data_name}_2009.csv")
df_all_seasons['Season'] = 2009
for season in range(2010, 2022):
    df_season = pd.read_csv(f"{var.path_to_data}/{var.ready_data_name}_{season}.csv")
    df_season['Season'] = season
    df_all_seasons = pd.concat([df_all_seasons, df_season], ignore_index=True)
df_all_seasons.to_csv(f"{var.path_to_data}/{var.ready_data_name}_all.csv", index=False)
