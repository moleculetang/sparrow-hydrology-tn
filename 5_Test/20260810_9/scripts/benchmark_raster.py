from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
PATH = Path(r"E:\SPARROW\0_reach_topology\results\rasters\reach_catchments.tif")


def main() -> None:
    started = perf_counter()
    with Image.open(PATH) as image:
        width, height = image.size
        count = 0
        per_id: dict[int, int] = {}
        for top in range(0, height, 64):
            arr = np.asarray(image.crop((0, top, width, min(top + 64, height))))
            valid = arr[arr > 0]
            count += int(valid.size)
            if valid.size:
                keys, values = np.unique(valid, return_counts=True)
                for key, value in zip(keys, values):
                    per_id[int(key)] = per_id.get(int(key), 0) + int(value)
            print(top, count, round(perf_counter() - started, 2), flush=True)
    print("done", width, height, count, len(per_id), round(perf_counter() - started, 2))


if __name__ == "__main__":
    main()
