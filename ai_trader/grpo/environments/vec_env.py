import multiprocessing as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    try:
        env = env_fn_wrapper()
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    # Automatically reset
                    terminal_ob = ob
                    ob, info_reset = env.reset()
                    info['reset_info'] = info_reset
                    info['terminal_observation'] = terminal_ob # Save for reference if needed
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send((ob, info))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        logger.info("Worker: KeyboardInterrupt")
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        logger.error(f"Worker Exception: {e}\n{err_msg}")
        remote.send(('exception', err_msg))
    finally:
        remote.close()


class SubprocVecEnv:
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    This effectively bypasses the python GIL.
    """
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # mp.Pipe creates a two-way connection
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(target=worker, args=(work_remote, remote, env_fn))
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
            self.ps.append(p)
            work_remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = []
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == 'exception':
                self.close()
                raise res[1]
            results.append(res)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
            
        results = []
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == 'exception':
                self.close()
                raise RuntimeError(f"Worker exception during reset: {res[1]}")
            results.append(res)
            
        obs, infos = zip(*results)
        return np.stack(obs), infos
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call a method of the wrapped environments."""
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            self.remotes[i].send(('env_method', (method_name, method_args, method_kwargs)))
        return [self.remotes[i].recv() for i in indices]

    def get_attr(self, attr_name, indices=None):
        """Read an attribute from the wrapped environments."""
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            self.remotes[i].send(('get_attr', attr_name))
        return [self.remotes[i].recv() for i in indices]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    if remote.poll(0.1): # 최대 0.1초 대기 (데드락 방지)
                        remote.recv()
                except Exception:
                    pass
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0) # 무한 대기 방지
            if p.is_alive():
                p.terminate() # 강제 종료
        self.closed = True

class DummyVecEnv:
    """단일 프로세스에서 동작하는 동기식 벡터 환경 구현체 (메모리 제약 / OOM 해결용)"""
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, d, tr, info = env.step(actions[i])
            if d or tr:
                terminal_ob = o
                o, reset_info = env.reset()
                info['reset_info'] = reset_info
                info['terminal_observation'] = terminal_ob
            obs.append(o)
            rews.append(r)
            dones.append(d or tr)
            infos.append(info)
        return np.stack(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        obs, infos = [], []
        for env in self.envs:
            o, i = env.reset()
            obs.append(o)
            infos.append(i)
        return np.stack(obs), infos

    def env_method(self, method_name, *args, **kwargs):
        results = []
        for env in self.envs:
            method = getattr(env, method_name)
            results.append(method(*args, **kwargs))
        return results

    def close(self):
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
