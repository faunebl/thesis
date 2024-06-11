import polars as pl 

def display_polars(frame: pl.DataFrame) -> None:
    with pl.Config(fmt_str_lengths= 100000, set_tbl_cols= -1, set_tbl_rows= -1):
        print(frame)
    return None