#!/usr/bin/env bash
#SBATCH --job-name=debug-nan
#SBATCH --account=aip-gigor
#SBATCH --partition=gpubase_l40s_b1,gpubase_l40s_b2
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-debug-nan-%j.out

set -euo pipefail
cd /project/aip-gigor/zhan5664/trajfm
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -c "
import torch, numpy as np

device = 'cuda'
torch.manual_seed(42)

# ---- Shared setup ----
from utils.hmt import H3Tokenizer, TimeFeatures
h3_tok = H3Tokenizer(res0=9, res1=7, res2=5, vocab_sizes=(16384, 4096, 1024), hash_tokens=True)

B, L = 4, 64
coords = torch.randn(B, L, 2) * 0.01
timestamps = torch.cumsum(torch.rand(B, L) * 30 + 1, dim=1)
mask = torch.ones(B, L)

# Tokenize
coords_np = coords.numpy()
l0, l1, l2 = [], [], []
for b in range(B):
    a, b_, c = h3_tok.tokenize(coords_np[b])
    l0.append(a); l1.append(b_); l2.append(c)
tokens_l0 = torch.from_numpy(np.stack(l0)).to(device)
tokens_l1 = torch.from_numpy(np.stack(l1)).to(device)
tokens_l2 = torch.from_numpy(np.stack(l2)).to(device)
timestamps = timestamps.to(device)
mask = mask.to(device)
coords = coords.to(device)

# Time features
time_enc = TimeFeatures(256).to(device)
time_embed = time_enc(timestamps, mask)
print(f'time_embed: nan={torch.isnan(time_embed).any()}, inf={torch.isinf(time_embed).any()}, max={time_embed.abs().max():.4f}')

# ---- Test PrivaMove (NO AMP) ----
print()
print('=== PrivaMove (no AMP, GPU float32) ===')
from utils.privamove import PrivaMoveModel, privamove_token_loss
from utils.flow import sample_rectified_flow_targets, flow_matching_loss
m_priv = PrivaMoveModel(embed_dim=256, depth=4, heads=8, num_cities=3, max_seq_len=64).to(device)

out = m_priv(tokens_l0, tokens_l1, tokens_l2, time_embed, mask)
print(f'  token_logits: nan={torch.isnan(out[\"token_logits\"]).any()}, max={out[\"token_logits\"].abs().max():.4f}')
print(f'  hidden: nan={torch.isnan(out[\"hidden\"]).any()}, max={out[\"hidden\"].abs().max():.4f}')

mask_idx = (torch.rand(B, L, device=device) < 0.3) & mask.bool()
loss_t = privamove_token_loss(out['token_logits'], tokens_l0, mask_idx, mask)
print(f'  token_loss: {loss_t.item():.4f} nan={torch.isnan(loss_t).any()}')

x_t, target_v, t_flow = sample_rectified_flow_targets(coords)
pred_v = m_priv.flow_head(out['hidden'], x_t, t_flow)
loss_f = flow_matching_loss(pred_v, target_v, mask)
print(f'  flow_loss: {loss_f.item():.4f} nan={torch.isnan(loss_f).any()}')

total = loss_t + 0.5 * loss_f
print(f'  total: {total.item():.4f}')
total.backward()
grad_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in m_priv.parameters())
print(f'  grad_nan: {grad_nan}')

# ---- Test PrivaMove after one step ----
opt = torch.optim.AdamW(list(m_priv.parameters()) + list(time_enc.parameters()), lr=1e-4)
opt.zero_grad()
total.backward() if not grad_nan else None
torch.nn.utils.clip_grad_norm_(m_priv.parameters(), 1.0)
opt.step()

time_embed2 = time_enc(timestamps, mask)
out2 = m_priv(tokens_l0, tokens_l1, tokens_l2, time_embed2, mask)
print(f'  After step1: token_logits nan={torch.isnan(out2[\"token_logits\"]).any()}, max={out2[\"token_logits\"].abs().max():.4f}')

# ---- Test Metabolic (with AMP) ----
print()
print('=== Metabolic (AMP, GPU) ===')
from utils.metabolic import MetabolicMobilityModel, metabolic_token_loss
m_met = MetabolicMobilityModel(embed_dim=256, depth=6, heads=8, max_seq_len=64).to(device)
modes = torch.randint(0, 6, (B, L), device=device)
speeds = torch.rand(B, L, device=device) * 10
hours = torch.rand(B, L, device=device) * 24
cong = torch.rand(B, L, device=device)
env = torch.randn(B, 24, 8, device=device)

with torch.amp.autocast(device_type='cuda'):
    out_m = m_met(tokens_l0, tokens_l1, tokens_l2, time_embed, mask, modes, speeds, hours, cong, env)
print(f'  token_logits: nan={torch.isnan(out_m[\"token_logits\"]).any()}, max={out_m[\"token_logits\"].abs().max():.4f}')
print(f'  hidden: nan={torch.isnan(out_m[\"hidden\"]).any()}, max={out_m[\"hidden\"].abs().max():.4f}')
print(f'  pred_emissions: nan={torch.isnan(out_m[\"pred_emissions\"]).any()}')

with torch.amp.autocast(device_type='cuda'):
    loss_mt = metabolic_token_loss(out_m['token_logits'], tokens_l0, mask_idx, mask)
print(f'  token_loss: {loss_mt.item():.4f} nan={torch.isnan(loss_mt).any()}')

# Without AMP
print()
print('=== Metabolic (NO AMP, GPU float32) ===')
out_m2 = m_met(tokens_l0, tokens_l1, tokens_l2, time_embed, mask, modes, speeds, hours, cong, env)
print(f'  token_logits: nan={torch.isnan(out_m2[\"token_logits\"]).any()}, max={out_m2[\"token_logits\"].abs().max():.4f}')
print(f'  hidden: nan={torch.isnan(out_m2[\"hidden\"]).any()}, max={out_m2[\"hidden\"].abs().max():.4f}')

# ---- Test SocialMob (with AMP) ----
print()
print('=== SocialMob (AMP, GPU) ===')
from utils.socialmob import SocialMobModel, socialmob_token_loss
m_soc = SocialMobModel(embed_dim=256, depth=6, heads=8, max_agents=8, max_steps_per_agent=32).to(device)
A = 4
coords_ma = torch.randn(B, A, 32, 2, device=device) * 0.01
ts_ma = torch.cumsum(torch.rand(B, A, 32, device=device) + 1, dim=2)
t0_ma = torch.randint(0, 100, (B, A, 32), device=device)
t1_ma = torch.randint(0, 100, (B, A, 32), device=device)
t2_ma = torch.randint(0, 100, (B, A, 32), device=device)
am = torch.ones(B, A, device=device)
sm = torch.ones(B, A, 32, device=device)
dt_ma = torch.ones(B, A, 32, device=device)
time_enc_ma = TimeFeatures(256).to(device)
flat_ts = ts_ma.reshape(B*A, 32)
flat_sm = sm.reshape(B*A, 32)
te_ma = time_enc_ma(flat_ts, flat_sm).reshape(B, A, 32, -1)

with torch.amp.autocast(device_type='cuda'):
    out_s = m_soc(t0_ma, t1_ma, t2_ma, te_ma, coords_ma, dt_ma, am, sm)
print(f'  token_logits: nan={torch.isnan(out_s[\"token_logits\"]).any()}, max={out_s[\"token_logits\"].abs().max():.4f}')
print(f'  hidden: nan={torch.isnan(out_s[\"hidden\"]).any()}')

# Without AMP
print()
print('=== SocialMob (NO AMP, GPU float32) ===')
out_s2 = m_soc(t0_ma, t1_ma, t2_ma, te_ma, coords_ma, dt_ma, am, sm)
print(f'  token_logits: nan={torch.isnan(out_s2[\"token_logits\"]).any()}, max={out_s2[\"token_logits\"].abs().max():.4f}')
print(f'  hidden: nan={torch.isnan(out_s2[\"hidden\"]).any()}')

print()
print('=== DIAGNOSIS COMPLETE ===')
"
