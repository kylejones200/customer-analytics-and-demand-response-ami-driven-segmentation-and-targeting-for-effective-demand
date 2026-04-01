# Customer Analytics and Demand Response: AMI-driven segmentation and targeting for effective demand... Electric grids must be built to meet peak demand even though those peaks
occur only a few times each year. This leads to expensive...

### Customer Analytics and Demand Response: AMI-driven segmentation and targeting for effective demand response for Electric Utilities
Electric grids must be built to meet peak demand even though those peaks
occur only a few times each year. This leads to expensive infrastructure
that sits underutilized for most hours. Demand response programs aim to
address this by encouraging customers to shift or reduce consumption
during critical periods, flattening peaks and easing stress on the grid.

However, not all customers respond the same way. Some households have
highly flexible loads they can shift easily, while others cannot reduce
usage without disruption. Programs that treat all customers alike often
achieve disappointing results because incentives and messaging fail to
match customer behavior.

Utilities need ways to identify which customers are best suited for
demand response and to design tailored programs that maximize
participation. Without this insight, they risk low enrollment, poor
event compliance, and limited grid impact, undermining the value of
demand-side resources.

### The Analytics Solution: Using Data to Target and Segment Customers
Customer analytics uses data from advanced metering infrastructure,
billing systems, and program participation records to understand
consumption patterns and identify load flexibility. Smart meters provide
granular consumption data that can reveal daily, weekly, and seasonal
usage profiles for each household.

Clustering techniques can segment customers into groups based on similar
load shapes. For example, some customers may show high evening peaks
tied to cooking and HVAC usage, while others have flatter profiles or
midday spikes associated with daytime occupancy. These segments inform
which customers are most likely to reduce load in response to incentives
or pricing signals.

Classification models can also predict participation likelihood, drawing
on historical demand response enrollment and demographic data. This
helps utilities prioritize outreach to those most likely to engage while
avoiding costly campaigns aimed at customers unlikely to respond.

By combining behavioral segmentation with predictive modeling, utilities
can refine demand response strategies, increase event performance, and
avoid overbuilding supply-side resources.

### Business Impact
Targeted demand response programs reduce peak load, defer costly
infrastructure upgrades, and lower wholesale energy procurement during
high-price periods. They also support integration of renewables by
shifting load to times of abundant solar or wind generation.

Customers benefit from lower bills through participation incentives or
time-based rates that reward off-peak consumption. Effective
segmentation ensures these programs feel relevant and fair, avoiding
customer dissatisfaction or program fatigue.

In competitive markets, successful demand response strategies can also
provide revenue streams by aggregating flexible load into virtual power
plants that bid into wholesale markets. Analytics makes this aggregation
more precise and dependable.

### Transition to the Demo
We will work with synthetic smart meter data to:

- Cluster customers into segments based on their daily load
  profiles.
- Visualize representative profiles for each segment to illustrate
  behavioral differences.
- Discuss how these segments inform targeted demand response design and
  incentive structures.

By the end, we will see how simple clustering techniques turn raw
consumption data into actionable customer insights, making demand
response programs more effective and efficient.

```python
"""
Customer Analytics and Demand Response
Load segmentation using smart meter data and clustering for DR targeting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans

def generate_smart_meter_data(customers=200, days=14):
    """
    Generate synthetic smart meter data (hourly kWh for multiple customers).
    """
    rng = pd.date_range("2022-01-01", periods=24 * days, freq="H")
    profiles = []
    for _ in range(customers):
        base = np.random.uniform(0.3, 1.0)  # Base consumption
        morning_peak = base + 0.5 * np.exp(-0.5 * (np.arange(24) - 7) ** 2)
        evening_peak = base + 0.8 * np.exp(-0.5 * (np.arange(24) - 19) ** 2)
        daily_profile = (morning_peak + evening_peak) / 2
        load = np.tile(daily_profile, days) + np.random.normal(0, 0.05, 24 * days)
        profiles.append(load)
    df = pd.DataFrame(profiles, columns=rng)
    return df

def cluster_load_profiles(df):
    """
    Cluster customer load profiles using KMeans on daily averages.
    """
    scaler = StandardScaler()
    daily_avg = df.values.reshape(df.shape[0], -1, 24).mean(axis=1)  # Average daily profile
    daily_scaled = scaler.fit_transform(daily_avg)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(daily_scaled)

    plt.figure(figsize=(10, 6))
    for c in range(3):
        cluster_profiles = daily_avg[labels == c]
        plt.plot(cluster_profiles.T, color="gray", alpha=0.2)
        plt.plot(cluster_profiles.mean(axis=0), label=f"Cluster {c+1}", linewidth=2)
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Load (kWh)")
    plt.title("Customer Segmentation for Demand Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter9_dr_clusters.png")
    plt.show()

    return labels

def identify_dr_targets(df, labels, target_cluster=2):
    """
    Identify customers in the highest-load cluster for DR targeting.
    """
    high_load_customers = np.where(labels == target_cluster)[0]
    print(f"Identified {len(high_load_customers)} customers for DR targeting.")
    return high_load_customers

if __name__ == "__main__":
    df_smart = generate_smart_meter_data()
    labels = cluster_load_profiles(df_smart)
    dr_targets = identify_dr_targets(df_smart, labels)
```


``` 
Identified 64 customers for DR targeting.
```
::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[October 5, 2025](https://medium.com/p/540059580b23).

[Canonical
link](https://medium.com/@kyle-t-jones/customer-analytics-and-demand-response-ami-driven-segmentation-and-targeting-for-effective-demand-540059580b23)

Exported from [Medium](https://medium.com) on November 10, 2025.
