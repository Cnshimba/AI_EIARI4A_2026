# Week 12: Real-Time Network Intrusion Detection on Jetson Orin Nano

> [!IMPORTANT]
> **Use Case**: You are building a "Network Sentry" device. You plug the Jetson into a switch port (or use it as a gateway), and it lights up red whenever it detects suspicious packet patterns.

## 1. The Strategy: Anomaly Detection
Unlike a firewall that looks for specific signatures (e.g., "Attack X"), we will build an **Anomaly Detector**.
1.  **Listen**: The Jetson listens to "Normal" traffic (YouTube, Email, Web Browsing) for 5 minutes.
2.  **Learn**: It trains a Neural Network (Autoencoder) to memorize this pattern.
3.  **Deploy**: It switches to "Guard Mode".
    *   If traffic looks like what it learned -> **Green Light**.
    *   If traffic looks completely different (e.g., Nmap Scan, DDoS) -> **Red Alarm**.

## 2. Prerequisites
On your Jetson, install the necessary Python libraries:
```bash
sudo pip3 install scapy pandas torch numpy
```

## 3. The "Guard Dog" Script
We have provided a unified script `Week_12_Jetson_IDS_Script.py` that handles all phases: **Capture**, **Train**, and **Monitor**.

### Phase A: Record "Normal" Baseline
Run the script in **capture mode**.
```bash
# Capture 1000 packets and save to 'normal_traffic.csv'
sudo python3 Week_12_Jetson_IDS_Script.py --mode capture --count 1000
```
*   *Activity*: While this is running, just browse the web normally on the Jetson. Do NOT run attacks yet.

### Phase B: Train the Brain
Now, train the Autoencoder on that data.
```bash
# Train the model and save it to 'ids_model.pth'
python3 Week_12_Jetson_IDS_Script.py --mode train
```
*   This uses PyTorch to create a model that compresses the network metadata and tries to reconstruct it. It learns what "normal" looks like.

### Phase C: Live Guard Mode (Active IPS)
Now, the fun part. Start the real-time blocking monitor.
```bash
sudo python3 Week_12_Jetson_IDS_Script.py --mode monitor
```
The screen will show a "heartbeat" of normal packets.

**Trigger an Attack:**
From another computer (or a different terminal), run an Nmap scan against the Jetson:
```bash
# Example attack
nmap -sS -p 1-1000 <JETSON_IP_ADDRESS>
```
1.  Watch the Jetson's terminal. It should scream **[ANOMALY DETECTED]**.
2.  It will then say **[IPS ACTION] Blocking IP...**.
3.  **Result**: The attacker's terminal will suddenly hang or timeout. You have effectively cut them off from the network!

> [!CAUTION]
> **Cleanup**: The script modifies the Jetson's system firewall. After you are done, **you must reset it** or you might remain blocked.
> Run this command to flush all rules:
> ```bash
> sudo iptables -F
> ```

## 4. How the Code Works (for your Report)
*   **Feature Extraction**: We use `scapy` to strip headers. We don't look at the *content* (encrypted payload), only the *metadata*:
    *   `Packet Size`: Attacks often have very small (scan) or very large (exfiltration) packets.
    *   `TTL` (Time To Live).
    *   `TCP Flags`: Syn floods look different from HTTP handshakes.
    *   `Ports`: We map Source/Dest ports to categories (Well-Known, Registered, Dynamic).
*   **The Model**: An **Autoencoder**.
    *   Input: Packet Features (Size, Ports, Flags...)
    *   Encoder: Compresses this to 3 numbers.
    *   Decoder: Tries to recreate the original input.
    *   **Logic**: If the Decoder fails (high error), it means the Encoder never saw this type of packet before -> **Anomaly**.
