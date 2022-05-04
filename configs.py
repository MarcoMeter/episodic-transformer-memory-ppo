def cartpole_masked_config():
    return {
        "env": "CartPoleMasked",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 100,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 4,
        "value_loss_coefficient": 0.2,
        "hidden_layer_size": 128,
        "learning_rate_schedule":
            {
            "initial": 3.0e-4,
            "final": 3.0e-6,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.0001,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 1000
            }
    }

def minigrid_config():
    return {
        "env": "Minigrid",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 500,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 8,
        "value_loss_coefficient": 0.25,
        "hidden_layer_size": 512,
        "learning_rate_schedule":
            {
            "initial": 2.0e-4,
            "final": 2.0e-4,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.001,
            "power": 1.0,
            "max_decay_steps": 300
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 300
            }
    }

def poc_memory_env_config():
    return {
        "env": "PocMemoryEnv",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 30,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 128,
        "n_mini_batch": 8,
        "value_loss_coefficient": 0.1,
        "hidden_layer_size": 64,
        "learning_rate_schedule":
            {
            "initial": 3.0e-4,
            "final": 3.0e-4,
            "power": 1.0,
            "max_decay_steps": 30
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.0001,
            "power": 1.0,
            "max_decay_steps": 30
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 30
            }
    }

def carpole_config():
    return {
        "env": "CartPole",
        "gamma": 0.99,
        "lamda": 0.95,
        "updates": 100,
        "epochs": 4,
        "n_workers": 16,
        "worker_steps": 256,
        "n_mini_batch": 4,
        "value_loss_coefficient": 0.2,
        "hidden_layer_size": 128,
        "learning_rate_schedule":
            {
            "initial": 3.0e-4,
            "final": 3.0e-6,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "beta_schedule":
            {
            "initial": 0.001,
            "final": 0.0001,
            "power": 1.0,
            "max_decay_steps": 100
            },
        "clip_range_schedule":
            {
            "initial": 0.2,
            "final": 0.2,
            "power": 1.0,
            "max_decay_steps": 1000
            }
    }