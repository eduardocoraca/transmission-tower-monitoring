from datetime import datetime
from pathlib import Path

SEQ_1_PATH = Path("./data/wpts/sequence_1_db35")
SEQ_2_PATH = Path("./data/wpts/sequence_2_db35")
SEQ_3_PATH = Path("./data/wpts/sequence_3_db35")

SEQ_1_HEALTH_END = datetime(2023, 1, 25, 10, 0, 0)
SEQ_2_HEALTH_END = datetime(2023, 5, 9, 11, 0, 0)
SEQ_3_HEALTH_END = datetime(2023, 8, 30, 11, 0, 0)


CABLES = ["c1", "c2", "c3", "c4"]
AXES = ["x", "y", "z"]
FS = 256
