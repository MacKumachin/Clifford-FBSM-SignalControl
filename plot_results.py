
import numpy as np
import matplotlib.pyplot as plt

def plot_rank_evolution():
    rank = np.load("rank_over_time.npy")
    plt.figure(figsize=(6, 4))
    plt.plot(rank, marker='o')
    plt.title("Surrogate Rank Evolution over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Surrogate Rank")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rank_evolution.png")
    plt.show()

def plot_final_curvature():
    curvature = np.load("final_curvature.npy")
    plt.figure(figsize=(6, 5))
    plt.imshow(curvature, cmap='viridis')
    plt.colorbar(label="Curvature Magnitude")
    plt.title("Final Curvature Field |Δf(x,T)|")
    plt.tight_layout()
    plt.savefig("final_curvature.png")
    plt.show()

if __name__ == "__main__":
    plot_rank_evolution()
    plot_final_curvature()
