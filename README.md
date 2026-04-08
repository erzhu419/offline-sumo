# offline-sumo

Offline-to-online RL for SUMO bus holding control.

## Status

Pure offline pretraining (AWR, 100K steps on `merged_all_v2.h5`) is done.
Pretrained checkpoint: `pretrained/offline_final.pt`

## Structure

```
agents/          h2oplus_bus.py, model.py
buffers/         bus_replay_buffer.py, mixed_replay_buffer.py
env/             sim_core/, envs/, sumo_env/, common/
utils/           bus_sampler.py, priority_index.py, snapshot_store.py
pretrained/      offline_final.pt (pretrained, 100K steps AWR)
data/            datasets_v2 symlink (~50GB, not tracked by git)
```

## Plan

- `train_offline_only.py`  — pure offline reference (done)
- `train_offline2online.py` — offline pretraining → online SUMO fine-tune (TODO)
