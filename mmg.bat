@echo off
set conda_path=C:\ProgramData\Anaconda3
set environment_name=episodic

call %conda_path%\Scripts\activate.bat %conda_path%
call conda activate %environment_name%

python train.py ^
  --env_id MortarMayhem-Grid-v0 ^
  --total_timesteps 75000000 ^
  --num_envs 16 ^
  --num_steps 512 ^
  --num_minibatches 8 ^
  --update_epochs 3 ^
  --trxl_num_blocks 3 ^
  --trxl_num_heads 4 ^
  --trxl_dim 384 ^
  --trxl_memory_length 119 ^
  --trxl_positional_encoding "absolute" ^
  --reconstruction_coef 0.0 ^
  --vf_coef 0.5 ^
  --max_grad_norm 0.5 ^
  --learning_rate 3.5e-4 ^
  --ent_coef 0.0001 ^
  --clip_coef 0.1
