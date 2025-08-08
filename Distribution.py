import os
import sys
import matplotlib.pyplot as plt


def draw_plots(df_name, df):
    labels = list(df.keys())
    values = list(df.values())
    fig, ax = plt.subplots(1, 2)

    ax[0].pie(
        values,
        autopct=lambda pct: f"{pct:.1f}%",
        colors=["blue", "red", "cyan", "purple"],
        textprops=dict(color="w"),
    )
    ax[0].set_title(f"{df_name} class distribution")

    ax[1].bar(labels, values, color=["blue", "red", "cyan", "purple"])
    ax[1].set_title(f"{df_name} class distribution")
    [e.set_rotation(45) for e in ax[1].get_xticklabels()]

    plt.tight_layout()
    plt.show()


def main(dir_path):
    df_name = dir_path[dir_path.rfind("/") + 1:]
    df = {}

    for sub_file in os.listdir(dir_path):
        if os.path.isdir(f"{dir_path}/{sub_file}"):
            df[sub_file] = len(os.listdir(f"{dir_path}/{sub_file}"))
        else:
            df[sub_file] = 1

    draw_plots(df_name, df)


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("Enter a directory as argument")
        if not os.path.isdir(sys.argv[1]):
            raise ValueError("The entered argument does not exist")
        if sys.argv[1].endswith("/"):
            sys.argv[1] = sys.argv[1][:-1]

        main(sys.argv[1])
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
