import subprocess

def get_node_info():
    result = subprocess.run(['sinfo', '-N', '-o', '%P %n %C %t'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Error running sinfo:", result.stderr)
        return []

    lines = result.stdout.strip().split('\n')[1:]  # skip header
    return lines

def parse_node_info(lines):
    partition_nodes = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        partition = parts[0].strip('*').lower()
        node = parts[1]
        cpus_str = parts[2]  # format: A/I/O/T
        state = parts[3].lower()

        try:
            alloc, idle, other, total = map(int, cpus_str.split('/'))
        except:
            continue

        if 'down' in state:
            continue

        if idle > 0 and 'gpu' not in partition:  # ❌ skip GPU partitions
            if partition not in partition_nodes:
                partition_nodes[partition] = []
            partition_nodes[partition].append({
                'node': node,
                'idle': idle,
                'total': total
            })
    return partition_nodes

def recommend_partition(partition_nodes, max_nodes=20):
    if not partition_nodes:
        print("❌ No CPU partitions with idle nodes found.")
        return None

    # Sort partitions by total idle CPUs
    sorted_partitions = sorted(
        partition_nodes.items(),
        key=lambda kv: sum(n['idle'] for n in kv[1]),
        reverse=True
    )

    for partition, nodes in sorted_partitions:
        if len(nodes) >= 1:
            count = min(max(len(nodes) - 1, 1), max_nodes)  # all available -1, but ≤ max_nodes
            return partition, nodes[:count]

    return None

def build_salloc_cmd(partition, nodes=1, ntasks=1, cpus=4, time="00:30:00", use_nodelist=False, nodelist=None):
    cmd = f"salloc --partition={partition} --nodes={nodes} --ntasks-per-node={ntasks} --cpus-per-task={cpus} --time={time}"
    if use_nodelist and nodelist:
        cmd += f" --nodelist={','.join(nodelist)}"
    return cmd

def main():
    cpus_per_task = 4
    ntasks_per_node = 1
    wall_time = "00:30:00"
    use_nodelist = False  # Only turn this on if you're sure pinning is needed
    max_nodes = 20        # ✅ Limit to 20 CPU nodes max

    print("🔍 Checking node-level SLURM CPU availability...\n")
    lines = get_node_info()
    partition_nodes = parse_node_info(lines)

    print("📊 CPU Partitions with idle nodes:")
    for p, nodes in partition_nodes.items():
        print(f"  {p:<10}: {len(nodes)} node(s) with idle CPUs")
        for n in nodes:
            print(f"    - {n['node']} | idle: {n['idle']} / total: {n['total']}")

    recommendation = recommend_partition(partition_nodes, max_nodes=max_nodes)
    if recommendation:
        partition, nodes = recommendation
        node_names = [n['node'] for n in nodes]

        print(f"\n✅ Recommended: Partition '{partition}' with {len(nodes)} node(s) (CPU only, capped at {max_nodes})")
        print(f"🧾 Nodes selected: {', '.join(node_names)}")

        print("\n💡 Suggested salloc:")
        print(build_salloc_cmd(
            partition=partition,
            nodes=len(nodes),
            ntasks=ntasks_per_node,
            cpus=cpus_per_task,
            time=wall_time,
            use_nodelist=use_nodelist,
            nodelist=node_names
        ))
    else:
        print("❌ No suitable CPU partition found with idle nodes.")

if __name__ == "__main__":
    main()
