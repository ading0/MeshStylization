import os
import pathlib
import zipfile

def is_zipped_dir(path: str) -> bool:
    return os.path.splitext(path)[1] == ".zip"

def examine_dir(dir: str) -> None:
    dir_path = os.path.abspath(dir)
    subs = [entry.path for entry in os.scandir(dir_path)]
    zip_dirs = filter(is_zipped_dir, subs)

    for zip_dr in zip_dirs:
        dst_dir = zip_dr[:-4]

        overwrite = os.path.exists(dst_dir)

        with zipfile.ZipFile(zip_dr, 'r') as zip_handle:
            zip_handle.extractall(dst_dir)
            if overwrite:
                print(f"Extracted (overwriting) {zip_dr} to {dst_dir}")
            else:
                print(f"Extracted {zip_dr} to {dst_dir}")

    subs = [entry.path for entry in os.scandir(dir_path)]
    sub_dirs = list(filter(os.path.isdir, subs))

    for sub_dir in sub_dirs: 
        examine_dir(sub_dir)

if __name__ == "__main__":
    examine_dir("./data/realistic")