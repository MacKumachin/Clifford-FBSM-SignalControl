import json
import numpy as np
from forward import solve_forward
from adjoint import solve_adjoint
from rank import compute_rank
import matplotlib.pyplot as plt

def main():
    # Load config
    with open("appl_II.json", "r") as f:
        config = json.load(f)

    T = config["T"]
    alpha = config["alpha"]
    beta = config["beta"]
    threshold = config["threshold"]

    # Initial conditions
    f0 = np.random.rand(32, 32)
    A = np.zeros_like(f0)
    u = np.zeros_like(f0)

    time_steps = 10
    rank_evolution = []

    f = f0
    for t in range(time_steps):
        f = solve_forward(f, A, u, T)
        lambda_ = solve_adjoint(f, A, T)
        u = (1 / (2 * beta)) * lambda_
        rank = compute_rank(f, threshold)
        rank_evolution.append(rank)

    # Save rank and curvature evolution
    np.save("rank_over_time.npy", np.array(rank_evolution))
    curvature_evolution = [np.linalg.norm(f_) for f_ in rank_evolution]
    np.save("curvature_over_time.npy", np.array(curvature_evolution))
    np.save("f_final.npy", f)

    # ✅ Curvature image from final f
    grad_x, grad_y = np.gradient(f)
    curvature = np.sqrt(grad_x**2 + grad_y**2)  # shape (32, 32)
    np.save("curvature_final.npy", curvature)

    # ✅ Plot curvature image
    plt.figure()
    plt.imshow(curvature, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("Curvature (Final)")
    plt.savefig("curvature_final.png")

    print("[main] Simulation complete. Output saved.")

if __name__ == "__main__":
    main()
