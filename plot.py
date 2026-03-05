#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_particles(input_data):
    df = pd.read_csv(input_data, dtype=np.float32)
    df.columns = df.columns.str.strip()
    max_clusterid = max(df["clusterId"])

    plt.figure(dpi=200)

    df_out = df[df.clusterId == -1]  # Outliers
    plt.scatter(df_out.x, df_out.y, s=10, marker="x", color="0.4")
    for i in range(0, int(max_clusterid) + 1):
        dfi = df[df.clusterId == i]  # ith cluster
        plt.scatter(dfi.x, dfi.y, s=10, marker=".")
    df_seed = df[df.isSeed == 1]  # Only Seeds
    plt.scatter(df_seed.x, df_seed.y, s=25, color="r", marker="*")

    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)

    plt.grid()

    plt.savefig("result.svg", format="svg", dpi=1200)

    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    if os.path.isfile(filename):
        plot_particles(filename)


if __name__ == "__main__":
    main()
