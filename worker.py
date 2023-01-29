import multiprocessing
import multiprocessing.connection
from utils import create_env

def worker_process(remote: multiprocessing.connection.Connection, config:dict) -> None:
    """Executes the threaded interface to the environment.
    
    Arguments:
        remote {multiprocessing.connection.Connection} -- Parent thread
        env_name {str} -- Name of the to be instantiated environment
    """
    # Spawn environment
    try:
        env = create_env(config)
    except KeyboardInterrupt:
        pass

    # Communication interface of the environment thread
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            else:
                raise NotImplementedError
        except:
            break

class Worker:
    """A worker that runs one environment on one thread."""
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    
    def __init__(self, env_name:str):
        """
        Arguments:
            env_name (str) -- Name of the to be instantiated environment
        """
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_name))
        self.process.start()