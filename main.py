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
    f0 = np.random.rand(32, 32)
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

    # Save curvature evolution
    curvature_evolution = [np.linalg.norm(f_) for f_ in rank_evolution]
    np.save("curvature_over_time.npy", np.array(curvature_evolution))

    # Save final state f
    np.save("f_final.npy", f)

# Save final curvature (make sure it is 2D)
curvature = np.abs(np.gradient(f)[0])  # shape (32,)
curvature_2d = curvature[:, np.newaxis]  # shape (32, 1)
np.save("curvature_final.npy", curvature_2d)

# Save final curvature image
import matplotlib.pyplot as plt

plt.figure()
if curvature_2d.ndim == 2:
    plt.imshow(curvature_2d, cmap="viridis", aspect="auto")
    plt.colorbar()
else:
    plt.plot(curvature_2d)
plt.title("Curvature (Final)")
plt.savefig("curvature_final.png")



    print("[main] Simulation complete. Output saved.")

if __name__ == "__main__":
    main()
