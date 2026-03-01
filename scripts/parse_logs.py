import re
import sys

def parse_log(filepath):
    # Regex to match the Iteration line
    # Iteration 1903/8000 (23.8%) | Timesteps: 3788100 | Mean Reward: 0.7965 | Win Rate: 21.9% | Trades: 24 | Sharpe: -0.05 | AvgHold: 0.6s | Policy Loss: 0.0003
    pattern = re.compile(
        r"Iteration (\d+)/.*?Mean Reward: ([-\d.]+) \| Win Rate: ([\d.]+)% \| Trades: (\d+) \| Sharpe: ([-\d.]+) \| AvgHold: ([\d.]+)s \| Policy Loss: ([-\d.]+)"
    )
    
    metrics = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if "Iteration" in line and "Mean Reward:" in line:
                match = pattern.search(line)
                if match:
                    it = int(match.group(1))
                    reward = float(match.group(2))
                    win_rate = float(match.group(3))
                    trades = int(match.group(4))
                    sharpe = float(match.group(5))
                    avg_hold = float(match.group(6))
                    policy_loss = float(match.group(7))
                    metrics.append({
                        'iteration': it,
                        'reward': reward,
                        'win_rate': win_rate,
                        'trades': trades,
                        'sharpe': sharpe,
                        'avg_hold': avg_hold,
                        'policy_loss': policy_loss
                    })
    
    if not metrics:
        print("No metrics found.")
        return

    print("## Training Logs Summary")
    print(f"Total Iterations Parsed: {len(metrics)} (from {metrics[0]['iteration']} to {metrics[-1]['iteration']})\n")
    
    print("| Iteration Range | Mean Reward | Win Rate (%) | Trades | Sharpe | AvgHold (s) | Policy Loss |")
    print("|-----------------|-------------|--------------|--------|--------|-------------|-------------|")
    
    # Calculate averages over buckets of 100 iterations
    bucket_size = 100
    current_bucket = []
    bucket_start = metrics[0]['iteration']
    
    for m in metrics:
        if m['iteration'] >= bucket_start + bucket_size:
            if current_bucket:
                avg_rew = sum(x['reward'] for x in current_bucket) / len(current_bucket)
                avg_win = sum(x['win_rate'] for x in current_bucket) / len(current_bucket)
                avg_trd = sum(x['trades'] for x in current_bucket) / len(current_bucket)
                avg_shp = sum(x['sharpe'] for x in current_bucket) / len(current_bucket)
                avg_hld = sum(x['avg_hold'] for x in current_bucket) / len(current_bucket)
                avg_pol = sum(x['policy_loss'] for x in current_bucket) / len(current_bucket)
                
                print(f"| {bucket_start:04d} - {m['iteration']-1:04d} | {avg_rew:11.4f} | {avg_win:12.2f} | {avg_trd:6.1f} | {avg_shp:6.3f} | {avg_hld:11.2f} | {avg_pol:11.4f} |")
                
            bucket_start = m['iteration'] - (m['iteration'] % bucket_size)
            if bucket_start == m['iteration']:
                 pass # Exact match, keep it
            else:
                 bucket_start = m['iteration']
            current_bucket = [m]
        else:
            current_bucket.append(m)

    # Last bucket
    if current_bucket:
        avg_rew = sum(x['reward'] for x in current_bucket) / len(current_bucket)
        avg_win = sum(x['win_rate'] for x in current_bucket) / len(current_bucket)
        avg_trd = sum(x['trades'] for x in current_bucket) / len(current_bucket)
        avg_shp = sum(x['sharpe'] for x in current_bucket) / len(current_bucket)
        avg_hld = sum(x['avg_hold'] for x in current_bucket) / len(current_bucket)
        avg_pol = sum(x['policy_loss'] for x in current_bucket) / len(current_bucket)
        end_it = metrics[-1]['iteration']
        print(f"| {bucket_start:04d} - {end_it:04d} | {avg_rew:11.4f} | {avg_win:12.2f} | {avg_trd:6.1f} | {avg_shp:6.3f} | {avg_hld:11.2f} | {avg_pol:11.4f} |")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_log(sys.argv[1])
