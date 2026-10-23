# -*- coding: utf-8 -*-
"""
动态显卡调度: 读一个任务文件 (每行一条 shell 命令, 用 {GPU} 占位), 哪张卡空就把任务派到哪张卡。
  空闲判定: nvidia-smi 报告的已用显存 < 3 GB, 且没有被本调度器派过任务 (锁文件 /tmp/t2g_gpu_<i>.lock)。
  任务结束后释放锁; 全部任务结束后退出。共用机器上别人随时可能占卡, 所以每次派任务前都重新探测。
用法: python gpu_queue.py jobs.txt [最多并行数]
"""
import os, sys, time, subprocess, shlex
jobs = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip() and not l.startswith("#")]
MAXP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOCK = "/tmp/t2g_gpu_%d.lock"


def free_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    res = []
    for ln in out.strip().splitlines():
        i, m = [x.strip() for x in ln.split(",")]
        if int(m) < int(os.environ.get("GPU_USED_MAX", "3000")) and not os.path.exists(LOCK % int(i)): res.append(int(i))   # GPU_USED_MAX: 已用显存上限 (MiB)
    return res


running = {}   # gpu -> (Popen, job)
print("任务 %d 条" % len(jobs), flush=True)
while jobs or running:
    for g, (p, j) in list(running.items()):
        if p.poll() is not None:
            print(time.strftime("%H:%M"), "GPU", g, "完成:", j[:70], "退出码", p.returncode, flush=True)
            os.remove(LOCK % g); del running[g]
    while jobs and len(running) < MAXP:
        fg = [g for g in free_gpus() if g not in running]
        if not fg: break
        g = fg[0]; j = jobs.pop(0); open(LOCK % g, "w").write(str(os.getpid()))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        running[g] = (subprocess.Popen(j.replace("{GPU}", str(g)), shell=True, env=env), j)
        print(time.strftime("%H:%M"), "GPU", g, "启动:", j[:70], flush=True); time.sleep(20)      # 等显存占上再探测下一张
    time.sleep(30)
print("QUEUE_DONE", flush=True)
