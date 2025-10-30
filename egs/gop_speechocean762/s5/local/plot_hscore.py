import json
import matplotlib.pyplot as plt


with open("/home/mcw/durga/pronunciation_scoring/Datasets/speechocean762/resource/scores_no_stress.json") as f:
    data = json.load(f)

phone_scores = []

for utterance in data.values():
    for word in utterance["words"]:
        phone_scores.extend(word["phones-accuracy"])

plt.hist(phone_scores, bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Human Phone-Level Accuracy Scores")
plt.xlabel("Accuracy Score")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig("/home/mcw/human_phone_accuracy_distribution.png")
