import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP

# ---------------------------------------------------------
# 1. Configuration & Feature Config
# ---------------------------------------------------------
MODEL_PATH = "ids_model.pth"
DATA_FILE = "normal_traffic.csv"
FEATURE_COLS = ['len', 'ttl', 'sport', 'dport', 'tcp_flags', 'proto_tcp', 'proto_udp']

# ---------------------------------------------------------
# 2. Neural Network Architecture (Autoencoder)
# ---------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder: Compress
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3) # Latent Space
        )
        # Decoder: Reconstruct
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ---------------------------------------------------------
# 3. Packet Feature Extractor
# ---------------------------------------------------------
def extract_features(packet):
    """ Turn a Scapy packet into a list of numbers """
    if not packet.haslayer(IP):
        return None
    
    # Defaults
    length = len(packet)
    ttl = packet[IP].ttl
    sport = 0
    dport = 0
    tcp_flags = 0
    proto_tcp = 0
    proto_udp = 0
    
    if packet.haslayer(TCP):
        sport = packet[TCP].sport
        dport = packet[TCP].dport
        # Simple flag mapping (SYN=2, ACK=16, etc.)
        tcp_flags = int(packet[TCP].flags)
        proto_tcp = 1
    elif packet.haslayer(UDP):
        sport = packet[UDP].sport
        dport = packet[UDP].dport
        proto_udp = 1
        
    # Normalization (Simple / Naive for demo purposes)
    # in a real system, you'd use a fitted scaler.
    # We divide by realistic max values to keep inputs between 0-1 essentially.
    
    return [
        length / 1500.0,       # Max MTU
        ttl / 255.0,           # Max TTL
        sport / 65535.0,       # Max Port
        dport / 65535.0,
        tcp_flags / 255.0,
        proto_tcp,
        proto_udp
    ]

# ---------------------------------------------------------
# 4. Mode: Capture (Data Collection)
# ---------------------------------------------------------
def run_capture(count):
    print(f"[*] Capturing {count} packets for training...")
    data = []
    
    def pkt_callback(pkt):
        feats = extract_features(pkt)
        if feats:
            data.append(feats)
            if len(data) % 100 == 0:
                print(f"Captured {len(data)}/{count}...")
            
    sniff(prn=pkt_callback, count=count, store=0)
    
    df = pd.DataFrame(data, columns=FEATURE_COLS)
    df.to_csv(DATA_FILE, index=False)
    print(f"[+] Saved baseline data to {DATA_FILE}")

# ---------------------------------------------------------
# 5. Mode: Train (Autoencoder)
# ---------------------------------------------------------
def run_train():
    print("[*] Loading data...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("Error: No data found. Run --mode capture first.")
        return

    # Convert to Tensor
    data_tensor = torch.FloatTensor(df.values)
    
    # Init Model
    model = Autoencoder(input_dim=len(FEATURE_COLS))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("[*] Training Autoencoder...")
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
    # Calculate Threshold (Mean Loss + 2 std devs) on training data
    with torch.no_grad():
        reconstructions = model(data_tensor)
        loss_vec = torch.mean((data_tensor - reconstructions)**2, dim=1)
        threshold = loss_vec.mean() + 2 * loss_vec.std()
        
    print(f"[+] Training Complete. Anomaly Threshold: {threshold.item():.6f}")
    
    # Save Model AND Threshold
    state = {'state_dict': model.state_dict(), 'threshold': threshold.item()}
    torch.save(state, MODEL_PATH)
    print(f"[+] Model saved to {MODEL_PATH}")

import subprocess

# ... (Previous imports remain, but subprocess is new)

# ---------------------------------------------------------
# 5.5 IPS Functionality (Active Blocking)
# ---------------------------------------------------------
blocked_ips = set()

def block_ip(ip_address):
    """ Executes an iptables command to DROP packets from this IP """
    if ip_address in blocked_ips:
        return

    print(f"\033[93m[IPS ACTION] Blocking IP: {ip_address}...\033[0m")
    try:
        # Check if already blocked (optional, but good for safety)
        # For this lab, we just run the append command.
        # Command: sudo iptables -A INPUT -s <IP> -j DROP
        subprocess.run(["iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"], check=True)
        blocked_ips.add(ip_address)
        print(f"\033[92m[SUCCESS] {ip_address} has been firewalled.\033[0m")
    except Exception as e:
        print(f"\033[91m[ERROR] Failed to block IP: {e}\033[0m")

# ---------------------------------------------------------
# 6. Mode: Monitor (Real-Time Inference)
# ---------------------------------------------------------
def run_monitor():
    print("[*] Loading Model...")
    try:
        checkpoint = torch.load(MODEL_PATH)
    except FileNotFoundError:
        print("Error: No model found. Run --mode train first.")
        return

    model = Autoencoder(input_dim=len(FEATURE_COLS))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    threshold = checkpoint['threshold']
    
    print(f"[*] Starting Monitor. Threshold: {threshold:.6f}")
    print("[!] Active IPS Mode: Anomalous IPs will be blocked.")
    print("Press Ctrl+C to stop.")
    
    def monitor_callback(pkt):
        feats = extract_features(pkt)
        if not feats:
            return
            
        tensor_in = torch.FloatTensor([feats])
        
        with torch.no_grad():
            reconstruction = model(tensor_in)
            loss = torch.mean((tensor_in - reconstruction)**2).item()
            
        if loss > threshold:
            src_ip = pkt[IP].src
            print(f"\033[91m[ANOMALY DETECTED] Loss: {loss:.5f} | Src: {src_ip} -> Dst: {pkt[IP].dst}\033[0m")
            
            # Active Defense: Block the Attacker
            block_ip(src_ip)
            
        else:
            # Print a dot just to show it's alive
            print(".", end="", flush=True)

    try:
        sniff(prn=monitor_callback, store=0)
    except KeyboardInterrupt:
        print("\nStopping.")
        print("[-] Note: Blocked IPs remain in iptables. Run 'sudo iptables -F' to clear them.")

# ---------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['capture', 'train', 'monitor'], required=True, help="Operating mode")
    parser.add_argument("--count", type=int, default=1000, help="Number of packets to capture (default: 1000)")
    
    args = parser.parse_args()
    
    if args.mode == 'capture':
        run_capture(args.count)
    elif args.mode == 'train':
        run_train()
    elif args.mode == 'monitor':
        run_monitor()
