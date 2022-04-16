python process.py -fold 0 \
                  -ids_fn "/mnt2/user15/merlot_r/merlot_reserve_backup/data/gsdata/yttemporal1b_ids_val.csv" \
                  -ids_fn_o "/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_log/video_log.csv"

python process.py -fold 1 \
                  -ids_fn "/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_log/video_log.csv" \
                  -ids_fn_o "/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_log/video_log_1.csv"
