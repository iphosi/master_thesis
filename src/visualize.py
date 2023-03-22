import pandas as pd
import os
import matplotlib.pyplot as plt
from natsort import natsorted
from statistics import mean


def vis_ppl(
    input_path="../evaluation/german-gpt2/sequential/leave_out",
    output_path="../evaluation/german-gpt2/sequential",
    pause=3
):
    dirs = natsorted(os.listdir(input_path))

    plt.figure(figsize=(12, 5))

    model_name = None
    tuning_method_list = []
    simple_ppl_list = []
    normal_ppl_list = []

    for name in dirs:
        file_path = os.path.join(input_path, name, "perplexity.csv")
        ppl_df = pd.read_csv(file_path)

        if model_name is None:
            model_name = ppl_df["Model Name"].iloc[0]
            tuning_method_list = ppl_df["Tuning Method"].values.tolist()

        simple_ppl_list.append(ppl_df["PPL Simple Text"].values.tolist())
        normal_ppl_list.append(ppl_df["PPL Normal Text"].values.tolist())

    simple_ppl_list = pd.DataFrame(simple_ppl_list).T.values.tolist()
    normal_ppl_list = pd.DataFrame(normal_ppl_list).T.values.tolist()

    for i, tuning_method in enumerate(tuning_method_list):
        simple_ppl = simple_ppl_list[i]
        normal_ppl = normal_ppl_list[i]

        plt.subplot(1, 2, 1)
        plt.plot(
            simple_ppl,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["ORIG", "FT"] else "solid"
        )
        plt.legend(loc="upper left", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel("Perplexity")
        plt.title("Simple Text Perplexity")

        plt.subplot(1, 2, 2)
        plt.plot(
            normal_ppl,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["ORIG", "FT"] else "solid"
        )
        plt.legend(loc="upper right", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel("Perplexity")
        plt.title("Normal Text Perplexity")

    plt.suptitle(model_name)
    plt.savefig(os.path.join(output_path, "perplexity.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


def vis_simp(
    input_path="../evaluation/german-gpt2/sequential/leave_out",
    output_path="../evaluation/german-gpt2/sequential",
    pause=3
):
    dirs = natsorted(os.listdir(input_path))

    plt.figure(figsize=(12, 5))

    model_name = None
    tuning_method_list = []
    fre_list = []
    word_log_freq_list = []

    for name in dirs:
        file_path = os.path.join(input_path, name, "simplicity.csv")
        simp_df = pd.read_csv(file_path)

        if model_name is None:
            model_name = simp_df["Model Name"].iloc[0]
            tuning_method_list = simp_df["Tuning Method"].values.tolist()

        fre_list.append(simp_df["Average FRE"].values.tolist())
        word_log_freq_list.append(simp_df["Average Word Log Frequency"].values.tolist())

    fre_list = pd.DataFrame(fre_list).T.values.tolist()
    word_log_freq_list = pd.DataFrame(word_log_freq_list).T.values.tolist()

    for i in range(2):
        fre_list[i] = [mean(fre_list[i])] * len(fre_list[i])
        word_log_freq_list[i] = [mean(word_log_freq_list[i])] * len(word_log_freq_list[i])

    for i, tuning_method in enumerate(tuning_method_list):

        fre = fre_list[i]
        word_log_freq = word_log_freq_list[i]

        plt.subplot(1, 2, 1)
        plt.plot(
            fre,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["ORIG", "FT"] else "solid"
        )
        plt.legend(loc="lower left", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel("Fre Score")
        plt.title("Readability")

        plt.subplot(1, 2, 2)
        plt.plot(
            word_log_freq,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["ORIG", "FT"] else "solid"
        )
        plt.legend(loc="lower left", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel("Word Log Frequency")
        plt.title("Lexical Simplicity")

    plt.suptitle(model_name)
    plt.savefig(os.path.join(output_path, "simplicity.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


def vis_sim(
    input_path="../evaluation/german-gpt2/sequential/similarity.csv",
    output_path="../evaluation/german-gpt2/sequential",
    pause=5
):
    sim_df = pd.read_csv(input_path)
    model_name = sim_df["Model Name"].iloc[0]

    plt.figure(figsize=(12, 5))

    for _, row in sim_df.iterrows():
        src_tgt = row.iloc[1]
        sim = row.iloc[2:].values
        layer = [i for i in range(len(sim))]

        plt.subplot(1, 2, 1)
        plt.plot(layer, sim, label=src_tgt)
        plt.xlabel("Layer")
        plt.ylabel("PPMCC")
        plt.legend(loc="lower left", prop={"size": 8})
        plt.title("Representational Similarity")

        plt.subplot(1, 2, 2)
        plt.plot(layer[:-1], sim[:-1], label=src_tgt)
        plt.xlabel("Layer")
        plt.ylabel("PPMCC")
        plt.legend(loc="lower left", prop={"size": 8})
        plt.title("Representational Similarity Drop Last")

    plt.suptitle(model_name)
    plt.savefig(os.path.join(output_path, "similarity.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


if __name__ == "__main__":
    vis_sim()
    vis_ppl()
    vis_simp()
    print("End")
