"""
train_cql.py
============
Conservative Q-Learning (CQL, Kumar et al. 2020) baseline for the SUMO
bus-holding offline benchmark.

Pure offline training (no online rollouts). Uses a twin-Q SAC backbone for
fairness with the SAC/WSRL/RLPD baselines, with the CQL pessimism term added
to the critic loss:

    L_Q = TD-MSE(Q1, target) + TD-MSE(Q2, target)
        + alpha_cql * (logsumexp_a Q(s, a') - Q(s, a_data))

The logsumexp is approximated by sampling N_act random actions per state plus
N_pi actions from the current policy; the sampled-Q values are concatenated
and passed through logsumexp. This is the standard CQL(H) variant.

Run:
    python train_cql.py --seed 42 --n_steps 100000 --device cuda --ckpt_every 10000
"""
import os, sys, csv, time, argparse, datetime
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in [_HERE, os.path.join(_HERE, "env"), os.path.join(_HERE, "agents"),
          os.path.join(_HERE, "buffers"), os.path.join(_HERE, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import (EmbeddingLayer, BusEmbeddingPolicy, BusEmbeddingQFunction,
                   BusSamplerPolicy, Scalar, soft_target_update)
from bus_replay_buffer import BusMixedReplayBuffer
from common.data_utils import set_route_length, build_edge_linear_map

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--device",       type=str,   default="cpu")
parser.add_argument("--n_steps",      type=int,   default=100000)
parser.add_argument("--batch_size",   type=int,   default=1024)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--alpha_cql",    type=float, default=1.0,
                    help="CQL pessimism weight (Kumar+2020 default 1.0--5.0)")
parser.add_argument("--n_actions",    type=int,   default=10,
                    help="number of random + policy actions sampled per state for logsumexp")
parser.add_argument("--ckpt_every",   type=int,   default=10000)
parser.add_argument("--dataset_file", type=str,
                    default=os.path.join(_HERE, "data", "datasets_v2", "merged_all_v2.h5"))
parser.add_argument("--no_eval", action="store_true",
                    help="skip during-training SUMO eval (faster training on GPU-only nodes)")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

_ts = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
out_dir = os.path.join(_HERE, "experiment_output", f"cql_seed{args.seed}_{_ts}")
os.makedirs(out_dir, exist_ok=True)
print(f"Output: {out_dir}")

# ── Route length (for state normalisation) ──────────────────────────────────
edge_xml = os.path.join(_HERE, "env", "network_data", "a_sorted_busline_edge.xml")
if os.path.exists(edge_xml):
    edge_map = build_edge_linear_map(edge_xml, "7X")
    route_length = max(edge_map.values()) if edge_map else 13119.0
else:
    route_length = 13119.0
set_route_length(route_length)

# ── Networks (twin Q + policy, same backbone as SAC/WSRL/RLPD) ──────────────
cat_cols = ["line_id", "bus_id", "station_id", "time_period", "direction"]
cat_code_dict = {
    "line_id":     {i: i for i in range(12)},
    "bus_id":      {i: i for i in range(389)},
    "station_id":  {i: i for i in range(1)},
    "time_period": {i: i for i in range(1)},
    "direction":   {0: 0, 1: 1},
}
obs_dim, action_dim, hidden_sz = 17, 2, 48
emb_tmpl = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = emb_tmpl.output_dim + (obs_dim - len(cat_cols))

policy = BusEmbeddingPolicy(state_dim, action_dim, hidden_sz,
                             emb_tmpl.clone(), action_range=1.0)
qf1 = BusEmbeddingQFunction(state_dim, action_dim, hidden_sz, emb_tmpl.clone())
qf2 = BusEmbeddingQFunction(state_dim, action_dim, hidden_sz, emb_tmpl.clone())
target_qf1 = deepcopy(qf1)
target_qf2 = deepcopy(qf2)

policy.to(args.device); qf1.to(args.device); qf2.to(args.device)
target_qf1.to(args.device); target_qf2.to(args.device)

policy_optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
qf_optim = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.lr)

# Automatic entropy tuning
log_alpha = Scalar(0.0).to(args.device)
alpha_optim = torch.optim.Adam(log_alpha.parameters(), lr=args.lr)
TARGET_ENTROPY = -float(action_dim)

DISCOUNT = 0.99
SOFT_TAU = 5e-3

# ── Load offline data ────────────────────────────────────────────────────────
print(f"Loading offline data: {args.dataset_file}")
buffer = BusMixedReplayBuffer(
    state_dim=obs_dim, action_dim=action_dim, context_dim=30,
    dataset_file=args.dataset_file, device=args.device,
    buffer_ratio=0.01,
    reward_scale=1.0, reward_bias=0.0,
)
print(f"Offline data: {buffer.fixed_dataset_size:,} transitions")

# ── Optional SUMO eval env ──────────────────────────────────────────────────
sampler_policy = BusSamplerPolicy(policy, args.device)
eval_sampler = None
if not args.no_eval:
    try:
        from envs.sumo_gym_env import SumoGymEnv
        from bus_sampler import BusEvalSampler
        _SUMO_DIR = os.path.join(os.path.dirname(_HERE), "sumo-rl",
                                  "_standalone_f543609", "SUMO_ruiguang", "online_control")
        if os.path.isdir(_SUMO_DIR):
            eval_env = SumoGymEnv(sumo_dir=_SUMO_DIR, edge_xml=edge_xml,
                                  max_steps=18000, line_id="7X")
            eval_sampler = BusEvalSampler(eval_env)
            print("SUMO eval enabled.")
        else:
            print("SUMO sim dir missing; running with --no_eval semantics.")
    except Exception as e:
        print(f"Could not initialise SUMO eval ({e}); training with no_eval semantics.")

# ── Training step ───────────────────────────────────────────────────────────
def cql_step(batch):
    obs       = batch["observations"]
    actions   = batch["actions"]
    rewards   = batch["rewards"].squeeze()
    next_obs  = batch["next_observations"]
    dones     = batch["dones"].squeeze().float()
    B = obs.shape[0]

    alpha = log_alpha().exp().detach()

    # ── Critic update ──
    with torch.no_grad():
        next_a, next_lp = policy(next_obs)
        next_q = torch.min(target_qf1(next_obs, next_a),
                           target_qf2(next_obs, next_a)).squeeze()
        td_target = rewards + (1.0 - dones) * DISCOUNT * (next_q - alpha * next_lp.squeeze())

    q1 = qf1(obs, actions).squeeze()
    q2 = qf2(obs, actions).squeeze()
    td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

    # ── CQL pessimism term: logsumexp over actions − Q on data ──
    # Sample N_act random actions in [-1, 1]^A
    N = args.n_actions
    rand_a = (torch.rand((N, B, action_dim), device=obs.device) * 2.0 - 1.0)
    # Sample N actions from current policy
    pi_a, pi_lp = policy(obs)
    pi_lp_flat = pi_lp.reshape(B)
    pi_a_repeat = pi_a.unsqueeze(0).expand(N, -1, -1).contiguous()
    pi_lp_repeat = pi_lp_flat.unsqueeze(0).expand(N, -1).contiguous()

    obs_repeat = obs.unsqueeze(0).expand(N, -1, -1).reshape(N * B, -1)
    rand_a_flat = rand_a.reshape(N * B, action_dim)
    pi_a_flat = pi_a_repeat.reshape(N * B, action_dim)

    # Q values on random actions and on policy actions
    q1_rand = qf1(obs_repeat, rand_a_flat).reshape(N, B)
    q2_rand = qf2(obs_repeat, rand_a_flat).reshape(N, B)
    q1_pi   = qf1(obs_repeat, pi_a_flat).reshape(N, B)
    q2_pi   = qf2(obs_repeat, pi_a_flat).reshape(N, B)

    # Importance weights for log-sum-exp (Kumar+2020 Eq. 4)
    log_unif = -float(action_dim) * np.log(2.0)  # uniform density in [-1,1]^A
    cat1 = torch.cat([
        q1_rand - log_unif,
        q1_pi   - pi_lp_repeat.detach(),
    ], dim=0)
    cat2 = torch.cat([
        q2_rand - log_unif,
        q2_pi   - pi_lp_repeat.detach(),
    ], dim=0)
    lse_q1 = torch.logsumexp(cat1, dim=0)  # [B]
    lse_q2 = torch.logsumexp(cat2, dim=0)

    cql_q1 = (lse_q1 - q1).mean()
    cql_q2 = (lse_q2 - q2).mean()
    cql_term = args.alpha_cql * (cql_q1 + cql_q2)

    q_loss = td_loss + cql_term
    qf_optim.zero_grad()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), 1.0)
    qf_optim.step()
    soft_target_update(qf1, target_qf1, SOFT_TAU)
    soft_target_update(qf2, target_qf2, SOFT_TAU)

    # ── Policy update ──
    pi_a2, pi_lp2 = policy(obs)
    q_pi = torch.min(qf1(obs, pi_a2), qf2(obs, pi_a2)).squeeze()
    policy_loss = (alpha * pi_lp2.squeeze() - q_pi).mean()
    policy_optim.zero_grad(); policy_loss.backward(); policy_optim.step()

    # ── Alpha update ──
    with torch.no_grad():
        _, pi_lp3 = policy(obs)
    alpha_loss = -(log_alpha() * (pi_lp3.detach().squeeze() + TARGET_ENTROPY)).mean()
    alpha_optim.zero_grad(); alpha_loss.backward(); alpha_optim.step()

    return dict(
        td_loss=td_loss.item(), cql_term=cql_term.item(),
        q_loss=q_loss.item(), policy_loss=policy_loss.item(),
        alpha=alpha.item(), mean_q=q1.mean().item(),
    )

# ── CSV logger ──────────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, "train_log.csv")
csv_f = open(csv_path, "w", newline="")
csv_w = csv.writer(csv_f)
csv_w.writerow(["step", "td_loss", "cql_term", "policy_loss", "alpha", "mean_q",
                "eval_return", "wall_sec"])

print(f"\nCQL training: {args.n_steps} steps, batch={args.batch_size}, "
      f"alpha_cql={args.alpha_cql}, n_actions={args.n_actions}")
t0 = time.time()
eval_return = 0.0

for step in range(1, args.n_steps + 1):
    batch = buffer.sample(args.batch_size, scope="real")
    info = cql_step(batch)

    if step % 1000 == 0:
        print(f"  [Step {step:6d}/{args.n_steps}]  "
              f"td={info['td_loss']:.2f}  cql={info['cql_term']:.2f}  "
              f"policy={info['policy_loss']:.2f}  α={info['alpha']:.3f}  "
              f"meanQ={info['mean_q']:.2f}  t={time.time()-t0:.0f}s")

    if eval_sampler is not None and step % 5000 == 0:
        trajs = eval_sampler.sample(sampler_policy, n_trajs=2, deterministic=True)
        eval_return = float(np.mean([sum(t["rewards"]) for t in trajs])) if trajs else 0.0
        print(f"     eval={eval_return:.1f}")

    csv_w.writerow([step, info['td_loss'], info['cql_term'], info['policy_loss'],
                    info['alpha'], info['mean_q'], eval_return, time.time() - t0])
    if step % 100 == 0:
        csv_f.flush()

    if step % args.ckpt_every == 0:
        ckpt_p = os.path.join(out_dir, f"checkpoint_step{step}.pt")
        torch.save(dict(policy=policy.state_dict(), qf1=qf1.state_dict(),
                        qf2=qf2.state_dict(), step=step), ckpt_p)

csv_f.close()
torch.save(dict(policy=policy.state_dict(), qf1=qf1.state_dict(),
                qf2=qf2.state_dict(), step=args.n_steps),
           os.path.join(out_dir, "model_final.pt"))
print(f"\nFinal model saved → {out_dir}/model_final.pt")
print(f"Done in {time.time()-t0:.0f}s")
