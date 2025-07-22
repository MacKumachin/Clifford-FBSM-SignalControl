
import json
import numpy as np
from forward import solve_forward
from adjoint import solve_adjoint
from rank import compute_rank

def main():
    # Load config
    with open("appl_II.json", "r") as f:
        config = json.load(f)

    T = config["T"]
    alpha = config["alpha"]
    beta = config["beta"]
    threshold = config["threshold"]

    # Dummy initial state, connection A, and control u
    f0 = np.random.randn(32, 32)
    A = np.zeros_like(f0)
    u = np.zeros_like(f0)

    time_steps = 10
    rank_evolution = []

    # Simulate FBSM over time
    f = f0
    for t in range(time_steps):
        f = solve_forward(f, A, u, T)
        lambda_ = solve_adjoint(f, A, T)
        u = (1 / (2 * beta)) * lambda_
        rank = compute_rank(f, threshold)
        rank_evolution.append(rank)

    # Save rank evolution
    np.save("rank_over_time.npy", np.array(rank_evolution))
Å@np.save("rank_over_time.npy", rank_list)
Å@np.save("f_final.npy", f)
Å@np.save("curvature_final.npy", curvature)

    # Save final curvature (dummy as Laplacian)
    curvature = np.abs(np.gradient(np.gradient(f)[0])[0])
    np.save("final_curvature.npy", curvature)

    print("[main] Simulation complete. Output saved.")

if __name__ == "__main__":
    main()
