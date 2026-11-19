from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional
from pydantic import BaseModel


class INCFile(BaseModel):
    sampled_at: datetime
    dt: float
    temperature: float
    x1: list[float]
    y1: list[float]
    x2: list[float]
    y2: list[float]

    @classmethod
    def from_txt(cls, path: Path) -> "INCFile":
        # 1) read raw data
        raw_data = pd.read_table(
            path, skiprows=9, sep="    ", header=None, engine="python"
        )
        raw_data.columns = ["x1", "y1", "x2", "y2"]

        # 2) read meta data
        with open(path) as h:
            for _ in range(8):
                content = next(h).strip("\n").strip(" ").lower()
                split_content = [c.strip(" ") for c in content.split("=")]
                if len(split_content) != 2:
                    continue
                name, data = split_content
                if name == "data":
                    sampled_at = datetime.strptime(data, "%d/%m/%Y %H:%M")
                elif name == "dt":
                    dt = float(data.split("[s]")[0])
                elif name == "temp":
                    temperature = float(data.split("[")[0])
                else:
                    continue
        return cls(
            sampled_at=sampled_at,
            dt=dt,
            temperature=temperature,
            x1=raw_data["x1"].to_list(),
            x2=raw_data["x2"].to_list(),
            y1=raw_data["y1"].to_list(),
            y2=raw_data["y2"].to_list(),
        )


class LMSFile(BaseModel):
    """
    A class to represent the data from an LMS file.

    Attributes:
    -----------
    temperature : float
        The temperature at the time of measurement.
    dt : float
        The time interval between measurements.
    sampled_at : datetime
        The datetime when the sample was taken.
    s4 : Optional[list[float]]
        Cable 4 tension over time.
    s3 : Optional[list[float]]
        Cable 3 tension over time.
    s2 : Optional[list[float]]
        Cable 2 tension over time.
    s1 : Optional[list[float]]
        Cable 1 tension over time.
    c4_z : Optional[list[float]]
        Cable 4 z-axis acceleration over time.
    c4_y : Optional[list[float]]
        Cable 4 y-axis acceleration over time.
    c4_x : Optional[list[float]]
        Cable 4 x-axis acceleration over time.
    c3_z : Optional[list[float]]
        Cable 3 z-axis acceleration over time.
    c3_y : Optional[list[float]]
        Cable 3 y-axis acceleration over time.
    c3_x : Optional[list[float]]
        Cable 3 x-axis acceleration over time.
    c2_z : Optional[list[float]]
        Cable 2 z-axis acceleration over time.
    c2_y : Optional[list[float]]
        Cable 2 y-axis acceleration over time.
    c2_x : Optional[list[float]]
        Cable 2 x-axis acceleration over time.
    c1_z : Optional[list[float]]
        Cable 1 z-axis acceleration over time.
    c1_y : Optional[list[float]]
        Cable 1 y-axis acceleration over time.
    c1_x : Optional[list[float]]
        Cable 1 x-axis acceleration over time.
    input_acc : Optional[list[float]]
        Input acceleration over time.
    input_force : Optional[list[float]]
        Input force over time.
    """

    temperature: float
    dt: float
    sampled_at: datetime
    s4: Optional[list[float]] = None
    s3: Optional[list[float]] = None
    s2: Optional[list[float]] = None
    s1: Optional[list[float]] = None
    c4_z: Optional[list[float]] = None
    c4_y: Optional[list[float]] = None
    c4_x: Optional[list[float]] = None
    c3_z: Optional[list[float]] = None
    c3_y: Optional[list[float]] = None
    c3_x: Optional[list[float]] = None
    c2_z: Optional[list[float]] = None
    c2_y: Optional[list[float]] = None
    c2_x: Optional[list[float]] = None
    c1_z: Optional[list[float]] = None
    c1_y: Optional[list[float]] = None
    c1_x: Optional[list[float]] = None
    input_acc: Optional[list[float]] = None
    input_force: Optional[list[float]] = None

    @classmethod
    def from_txt(cls, path: Path) -> "LMSFile":
        """
        Create an LMSFile instance from a text file.

        Parameters:
        -----------
        path : Path
            The path to the text file.

        Returns:
        --------
        LMSFile
            An instance of LMSFile populated with data from the text file.
        """
        # 1) Read raw data
        raw_data = pd.read_table(
            path, skiprows=9, sep="    ", header=None, engine="python"
        )
        raw_cols = (
            pd.read_table(path, skiprows=8)
            .iloc[0]
            .index.to_list()[0]
            .strip()
            .split("  ")
        )
        raw_data.columns = raw_cols

        # 2) Read metadata
        with open(path) as h:
            for _ in range(8):
                content = next(h).strip("\n").strip(" ").lower()
                split_content = [c.strip(" ") for c in content.split("=")]
                if len(split_content) != 2:
                    continue
                name, data = split_content
                if name == "data":
                    sampled_at = datetime.strptime(data, "%d/%m/%Y %H:%M")
                elif name == "dt":
                    dt = float(data.split("[s]")[0])
                elif name == "temp":
                    temperature = float(data.split("[")[0])
                else:
                    continue

        # handling incorrect column names
        s4_col = "S4_T2[N]" if "S4_T2[N]" in raw_data.columns else "S4_T4[N]"

        return cls(
            temperature=temperature,
            dt=dt,
            sampled_at=sampled_at,
            s4=raw_data[s4_col].to_list(),
            s3=raw_data["S3_T2[N]"].to_list(),
            s2=raw_data["S2_T2[N]"].to_list(),
            s1=raw_data["S1_T2[N]"].to_list(),
            c4_z=raw_data["c4t2_z:+Z[g]"].to_list(),
            c4_y=raw_data["c4t2_y:+Y[g]"].to_list(),
            c4_x=raw_data["c4t2_x:+X[g]"].to_list(),
            c3_z=raw_data["c3t2_z:+Z[g]"].to_list(),
            c3_y=raw_data["c3t2_y:+Y[g]"].to_list(),
            c3_x=raw_data["c3t2_x:+X[g]"].to_list(),
            c2_z=raw_data["c2t2_z:+Z[g]"].to_list(),
            c2_y=raw_data["c2t2_y:+Y[g]"].to_list(),
            c2_x=raw_data["c2t2_x:+X[g]"].to_list(),
            c1_z=raw_data["c1t2_z:+Z[g]"].to_list(),
            c1_y=raw_data["c1t2_y:+Y[g]"].to_list(),
            c1_x=raw_data["c3t2_x:+X[g]"].to_list(),
            input_acc=raw_data["Input_T2_Acc:+X[g]"].to_list(),
            input_force=raw_data["Input_T2_F:+X[N]"].to_list(),
        )
