# VAE Full128 Fidelity Runbook (on5/on4)

Runbook for starting VAE training with `config_VAE_full128_fidelity_on5.json` on the MAISI server.

## 1) Open a terminal and SSH

```bash
ssh mhuber@on5.cir.meduniwien.ac.at
```

If on5 is busy, use on4:

```bash
ssh mhuber@on4.cir.meduniwien.ac.at
```

## 2) Go to the MAISI repo

```bash
cd /optima/exchange/mhuber/new_git/maisi/
```

## 3) Start tmux (recommended)

```bash
tmux new -s maisi_vae
```

Reconnect later with:

```bash
tmux attach -t maisi_vae
```

## 4) Allocate a GPU

```bash
srun -n16 --mem=50G --qos=longrunning --time=12-12:00:00 --gres=gpu:1 --nodelist=on5 -p full_optima -J "vae_full128_fidelity" --pty /bin/bash
```

Alternative node:

```bash
srun -n16 --mem=50G --qos=longrunning --time=12-12:00:00 --gres=gpu:1 --nodelist=on4 -J "vae_full128_fidelity" --pty /bin/bash
```

## 5) Build and enter the Singularity image

Only needed if the image is not already built:

```bash
sudo singularity build maisi.sif maisi.def
```

Enter the container:

```bash
singularity shell --nv maisi.sif
```

## 6) Start VAE training

```bash
python train_vae.py --config ./configs/config_VAE_full128_fidelity_on5.json
```

## 7) Encode images to latents (after training)

Use the actual run directory produced by training (timestamped). Example:

```bash
python ./scripts/encode_to_latents.py \
  --input_dir "/optima/exchange/mhuber/KermanyV3_resized/train" \
  --output_dir "./outputs/latents_full128_fidelity" \
  --autoencoder_path "./runs/VAE/vae_full128_fidelity_on5_YYYYMMDD_HHMM/model_best.pt" \
  --vae_config "./configs/config_VAE_full128_fidelity_on5.json"
```

Notes:
- Replace `YYYYMMDD_HHMM` with your actual run folder name.
- If a `*_best.pt` symlink exists but is broken, use the direct path to `model_best.pt` inside the run folder.

## Notes on the config

- Config file: `configs/config_VAE_full128_fidelity_on5.json`
- Dataset path: `/optima/exchange/mhuber/KermanyV3_resized/train`
- Output dir: `./runs/vae_full128_fidelity_on5`
- Key settings: batch size 32, lr 1e-4, epochs 100, L1 recon loss, AMP on

If you want to point to a different dataset, update `data.image_dir` in the config.
