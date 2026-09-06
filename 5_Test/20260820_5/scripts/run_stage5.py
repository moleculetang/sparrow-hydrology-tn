import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"E:\SPARROW\5_Test\20260820_4\scripts")))
from registered_stop import write_registered_stop

if __name__ == "__main__":
    write_registered_stop("20260820_5", "aquatic temperature sensitivity versus matched AQ_NULL")

