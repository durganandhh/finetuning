import pandas as pd
import matplotlib.pyplot as plt

# Read CSV, treat everything as string
df = pd.read_csv("/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/eer_results.csv")

# Convert numeric EERs to float, NA/All Correct → NaN
df['EER_numeric'] = pd.to_numeric(df['EER(%)'], errors='coerce')

# Plot histogram of numeric EERs only
plt.hist(df['EER_numeric'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.xlabel("EER (%)")
plt.ylabel("Number of Utterances")
plt.title("Distribution of numeric EER across utterances")
plt.show()
