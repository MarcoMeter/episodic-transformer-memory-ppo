@echo off
set conda_path=C:\ProgramData\Anaconda3
set environment_name=episodic

call %conda_path%\Scripts\activate.bat %conda_path%
call conda activate %environment_name%

python train.py ^
  --env_type PocMemoryEnv ^
  --total_timesteps 25000 ^
  --num_envs 16 ^
  --num_steps 128 ^
  --num_minibatches 8 ^
  --update_epochs 4 ^
  --trxl_num_blocks 4 ^
  --trxl_num_heads 1 ^
  --trxl_dim 64 ^
  --trxl_memory_length 32 ^
  --trxl_positional_encoding none ^
  --vf_coef 0.1 ^
  --max_grad_norm 0.5 ^
  --learning_rate 3.0e-4 ^
  --ent_coef 0.001 ^
  --clip_coef 0.2