import torch
from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser(options["--config"]).get_config()

    # Determine the device to be used for training and set the default tensor type
    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()