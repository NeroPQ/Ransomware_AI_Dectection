import random
from pathlib import Path
from uuid import uuid4
import pandas as pd
from datetime import datetime, timedelta

def main():
    random.seed(42)

    events = []

    base_time = datetime.now().replace(microsecond=0)
    event_id = 0

    def add_event(ts, process, pid, parent, event_type, operation, obj, label, family="benign",
                  bytes_written=0, entropy_before=0.0, entropy_after=0.0, dst_ip="", dst_port=0, domain=""):
        nonlocal event_id
        events.append({
            "event_index": event_id,
            "timestamp": ts.isoformat(),
            "source_type": "simulated_behavior",
            "host_id": "lab-host-01",
            "process_name": process,
            "pid": pid,
            "parent_process": parent,
            "event_type": event_type,
            "operation": operation,
            "object": obj,
            "bytes_written": bytes_written,
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "dst_ip": dst_ip,
            "domain": domain,
            "label": label,
            "family": family
        })
        event_id += 1

    # Benign phase
    benign_processes = [
        ("chrome", 1100, "systemd"),
        ("code", 1200, "systemd"),
        ("python", 1300, "bash"),
        ("libreoffice", 1400, "systemd")
    ]

    for i in range(1800):
        ts = base_time + timedelta(seconds=i)
        proc, pid, parent = random.choice(benign_processes)
        r = random.random()

        if r < 0.45:
            f = f"/home/user/docs/doc_{random.randint(1,120)}.txt"
            add_event(ts, proc, pid, parent, "file_read", "read", f, "benign")
        elif r < 0.68:
            f = f"/home/user/docs/doc_{random.randint(1,120)}.txt"
            add_event(ts, proc, pid, parent, "file_write", "write", f, "benign",
                       bytes_written=random.randint(100, 5000),
                       entropy_before=random.uniform(3.0, 5.0),
                       entropy_after=random.uniform(3.0, 5.5))
        elif r < 0.78:
            add_event(ts, proc, pid, parent, "process_create", "exec", proc, "benign")
        elif r < 0.90:
            domain = random.choice(["ubuntu.com", "github.com", "example.com", "pypi.org"])
            add_event(ts, proc, pid, parent
                      , "dns_query", "query", domain, "benign", domain=domain)
        else:
            add_event(ts, proc, pid, parent, "network_connect", "connect", "remote", "benign",
                      dst_ip=f"93.184.216.{random.randint(1, 200)}",
                      dst_port=random.choice([80, 443]))

    # Ransomware-like simulated phase
    sim_proc = "suspicious_sim.exe"
    pid = 6666
    parent = "bash"

    for i in range(900):
        ts = base_time + timedelta(seconds=1800 + i // 8)
        phase = i

        if phase < 100:
            f = f"/home/user/docs/doc_{random.randint(1, 120)}.txt"
            add_event(ts, sim_proc, pid, parent, "file_read", "read", f, "ransomware", "SimRansom")
        elif phase < 720:
            doc_id = random.randint(1, 250)
            if random.random() < 0.55:
                f = f"/home/user/docs/doc_{doc_id}.txt"
                add_event(ts, sim_proc, pid, parent, "file_write", "write", f, "ransomware", "SimRansom",
                          bytes_written=random.randint(5000, 70000),
                          entropy_before=random.uniform(3.0, 5.0),
                          entropy_after=random.uniform(7.0, 8.0))
            else:
                f = f"/home/user/docs/doc_{doc_id}.txt"
                add_event(ts, sim_proc, pid, parent, "file_read", "read", f, "ransomware", "SimRansom",
                          entropy_before=random.uniform(3.0, 5.0),
                          entropy_after=random.uniform(3.0, 5.0))
        elif phase == 720:
            add_event(ts, sim_proc, pid, parent, "shadow_copy_delete", "delete", "shadow_copy", "ransomware", "SimRansom")
        elif 721 <= phase < 760:
            add_event(ts, sim_proc, pid, parent, "service_stop", "stop", random.choice(["backup", "security", "database"]), "ransomware", "SimRansom")
        elif 760 <= phase < 820:
            add_event(ts, sim_proc, pid, parent, "registry_set", "set", "/registry/run/key", "ransomware", "SimRansom")
        else:
            domain = random.choice(["random-check.example", "update-node.example", "sync-api.example"])
            add_event(ts, sim_proc, pid, parent, "dns_query", "query", domain, "ransomware", "SimRansom", domain=domain)

    df = pd.DataFrame(events)
    output = Path("data/raw/behavior/safe_behavior.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df["session_id"] = uuid4().hex
    df.to_csv(output, index=False)

    print("[+] Created data/raw/behavior/safe_behavior.csv")
    print(df.shape)
    print(df["label"].value_counts())
    print(df["event_type"].value_counts())

if __name__ == "__main__":
    main()
