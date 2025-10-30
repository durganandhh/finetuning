import matplotlib.pyplot as plt

# Parse GOP scores from gop.1.txt
gop_scores = []

with open("/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/exp/gop_test/gop.1.txt") as f:
    for line in f:
        parts = line.strip().split("] [")
        for part in parts:
            try:
                score = float(part.strip("[]").split()[1])
                gop_scores.append(score)
            except:
                continue

print(f"Total GOP scores parsed: {len(gop_scores)}")
print("Sample GOP scores:", gop_scores[:10])
print("Min GOP score:", min(gop_scores))
print("Max GOP score:", max(gop_scores))


plt.hist(gop_scores, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of All GOP Scores")
plt.xlabel("GOP Score")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig("/home/mcw/gop_score_distribution.png")
# plt.show() 
