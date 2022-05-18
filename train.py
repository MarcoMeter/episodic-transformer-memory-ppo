import torch
from configs import cartpole_config, cartpole_masked_config, minigrid_config, poc_memory_env_config
from docopt import docopt
from trainer import PPOTrainer

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(cartpole_masked_config(), run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()