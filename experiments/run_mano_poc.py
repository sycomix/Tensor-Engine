import os
import sys

# Add project root to sys.path
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensor_engine as te
import argparse
from scripts.tasks.mano import ManoEmulator, generate_random_program, ManoState
from scripts.eval_harness import ExperimentLogger


def state_to_vector(s: ManoState, float=None, float=None, float=None, float=None, float=None, float=None) -> np.ndarray:
    """Features: [AC, PC, IR, DR, E, I, S, FGI, FGO, IEN] -> 10 floats."""
    scale = 1.0 / 65535.0
    vec = np.array([
        s.AC * scale,
        s.PC * scale,
        s.IR * scale,
        s.DR * scale,
        float(s.E),
        float(s.I),
        float(s.S),
        float(s.FGI),
        float(s.FGO),
        float(s.IEN)
    ], dtype=np.float32)
    return vec


def vector_to_next_ac_target(s_next: ManoState, float=None) -> np:
    return float(s_next.AC) / 65535.0


def run_mano_poc(args, int=None, int=None, abs=None, abs=None, len=None, range=None, enumerate=None, len=None,
                 range=None, len=None, len=None, range=None, range=None):
    # 1. Setup
    logger = ExperimentLogger(args.log_dir, "mano_nl_oob_stage2")
    logger.info("Starting Mano Experiment (Recreating Paper 2510.25741 with NL-OOB + Stage-II Loss)")

    # 2. Config
    d_model = 64
    d_ff = 128
    num_heads = 4
    t_max = 8
    beta = 0.05  # Entropy regularization strength
    nl_oob_config = "logarithmic"  # KEY: Out-of-Band Non-Linearity
    nl_oob_scale = 5.0

    logger.info(f"Config: d_model={d_model}, t_max={t_max}, beta={beta}, nl_oob={nl_oob_config}")

    # 3. Model
    input_dim = 10
    input_proj = te.Linear(input_dim, d_model, True)

    # Looped Transformer with NL-OOB
    lt = te.LoopedTransformer(
        d_model, d_ff, num_heads,
        nl_oob_config, nl_oob_scale,
        t_max, beta
    )

    output_proj = te.Linear(d_model, 1, True)

    # Optimizer
    params = input_proj.parameters() + lt.parameters() + output_proj.parameters()
    opt = te.Adam(lr=5e-4, beta1=0.9, beta2=0.999, eps=1e-8)

    mse_fn = te.MSELoss()

    batch_size = 32
    updates = 500

    # 4. Data Generation
    logger.info("Generating Training Traces...")
    transitions = []
    # Generate enough data
    for _ in range(200):
        prog, _ = generate_random_program(length=30)
        emu = ManoEmulator()
        emu.load_program(prog)
        trace = emu.run(max_steps=60)
        for i in range(len(trace) - 1):
            transitions.append((state_to_vector(trace[i]), vector_to_next_ac_target(trace[i + 1])))

    logger.info(f"Dataset Size: {len(transitions)}")

    # 5. Training Loop
    logger.info("Starting Training...")

    for step in range(updates):
        # Batching
        indices = np.random.choice(len(transitions), batch_size)
        x_batch = np.array([transitions[i][0] for i in indices])
        y_batch = np.array([transitions[i][1] for i in indices]).reshape(batch_size, 1)

        t_x = te.Tensor(x_batch.flatten().tolist(), [batch_size, input_dim])
        t_y = te.Tensor(y_batch.flatten().tolist(), [batch_size, 1])

        opt.zero_grad(params)

        # Forward
        emb = input_proj.forward(t_x).reshape([batch_size, 1, d_model])
        dist = te.Tensor([0.0], [1, 1])  # dummy distance

        # Looped Forward -> (outs: List[Tensor], p_phi: Tensor[B, t_max])
        outs, p_phi = lt.forward_looped(emb, dist)

        # Compute losses for EACH step
        step_losses_list = []
        for i, out_t in enumerate(outs):
            # out_t: [B, 1, D] -> [B, D]
            flat = out_t.reshape([batch_size, d_model])
            pred = output_proj.forward(flat)  # [B, 1]

            # Per-sample loss: (pred - y)^2
            # MSELoss typically returns MEAN. We need [B, 1] or [B] scalar losses.
            # tensor_engine's MSELoss returns SCALAR mean.
            # We need element-wise loss.
            # (pred - y).pow(2)
            # Check availability of ops. 'sub', 'pow'?
            diff = pred.sub(t_y)
            sq_diff = diff.pow(2.0)  # [B, 1]

            # Flatten to [B]
            sq_diff_flat = sq_diff.reshape([batch_size])
            step_losses_list.append(sq_diff_flat)

        # Stack to [B, t_max]
        # Using specific 'axis=1' to stack columns -> [B, T]
        step_losses = te.py_stack(step_losses_list, 1)

        # Stage-II Gate Loss
        # L = mean_B( sum_t(survival_t * step_loss_t) - beta * H(p) )
        loss = lt.stage2_loss(p_phi, step_losses)

        loss.backward()
        opt.step(params)

        if step % 50 == 0:
            logger.log_metric(step, {"stage2_loss": loss.get_data()[0]})

    # 6. Eval (Simulated 500 steps)
    logger.info("Evaluation...")
    test_prog, _ = generate_random_program(length=20, seed=123)
    emu_test = ManoEmulator()
    emu_test.load_program(test_prog)
    trace = emu_test.run(max_steps=50)

    total_err = 0.0
    count = 0
    correct_approx = 0

    for i in range(len(trace) - 1):
        x = state_to_vector(trace[i])
        y_gt = vector_to_next_ac_target(trace[i + 1])

        t_x = te.Tensor(x.flatten().tolist(), [1, input_dim])
        emb = input_proj.forward(t_x).reshape([1, 1, d_model])
        dist = te.Tensor([0.0], [1, 1])

        outs, p_phi = lt.forward_looped(emb, dist)

        # In inference, we take the weighted average, or the last step?
        # The paper suggests taking E[y_t] via p_phi, or argmax.
        # Let's take the Last Step for simplicity in metric, or Weighted?
        # Weighted:
        # preds = [proj(o) for o in outs] -> stack -> [B, T]
        # result = sum(p_phi * preds)
        # Let's verify 'last step' performance first as "pondering complete".

        pred = output_proj.forward(outs[-1].reshape([1, d_model])).get_data()[0]

        err = abs(pred - y_gt)
        total_err += err

        if abs(int(pred * 65535) - int(y_gt * 65535)) < 5:
            correct_approx += 1

        count += 1

    avg_err = total_err / count
    acc = correct_approx / count
    logger.log_result({
        "test_mae": avg_err,
        "test_acc": acc
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    if not hasattr(te, 'LoopedTransformer'):
        print("Error: Library missing LoopedTransformer.")
        sys.exit(1)

    run_mano_poc(args)
