import numpy as np

# Apple stock prices for 10 days
prices = np.array([182.5, 183.2, 181.9, 184.0, 185.1,
                   183.8, 186.2, 187.0, 185.5, 188.0])

# Calculate daily returns
returns = (prices[1:] - prices[:-1]) / prices[:-1]

# Print with labels
for i, r in enumerate(returns):
    print(f"Day {i+1} → Day {i+2}: {r*100:.2f}%")

# Calculate cumulative returns
cumulative_returns = np.prod(1 + returns) - 1
print(f"\nAverage daily return : {returns.mean()*100:.2f}%")
print(f"Cumulative return : {cumulative_returns*100:.2f}%")
print(f"std deviation (risk):{returns.std()*100:.2f}%")
print(f"Best day return: {returns.max()*100:.2f}%")
print(f"Worst day return: {returns.min()*100:.2f}%")