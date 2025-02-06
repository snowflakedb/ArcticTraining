import sys

all_reduce = "/code/users/jrasley/all_reduce_bench.py"

# Check if the hostfile path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <hostfile>")
    sys.exit(1)

hostfile_path = sys.argv[1]

# Read the hostfile and extract IP addresses
with open(hostfile_path, 'r') as f:
    hostfile_lines = f.readlines()

# Extract IP addresses
ip_addresses = [line.split()[0] for line in hostfile_lines]
if len(ip_addresses) % 2 != 0:
  print(f'dropping {ip_addresses[-1]}')
  ip_addresses = ip_addresses[:-1]

# Create pairs of IPs sequentially without overlap
ip_pairs = []
for i in range(0, len(ip_addresses), 2):
    ip1 = ip_addresses[i]
    ip2 = ip_addresses[i + 1]
    ip_pairs.append(f"{ip1}@{ip2}")

print('mkdir -p results')
# Print the list of IP pairs
for i, pair in enumerate(ip_pairs):
    print(f"deepspeed -H {hostfile_path} -i {pair} {all_reduce} &> results/{i}.log &")