#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastembed"]
# ///
"""Generate real embeddings for the auction simulation.

Uses BAAI/bge-small-en-v1.5 (384D, open-weight) via fastembed (ONNX runtime).
Outputs embeddings.go with advertiser and query vectors as Go literals.

Usage:
    cd /Users/junekim/Documents/openauction
    uv run cmd/simulate/gen_embeddings.py
"""

import numpy as np
from fastembed import TextEmbedding

# --- Advertisers ---
# Each entry: (name, description for embedding, max_value, base_bid, base_sigma, base_budget, strategy)
# Strategy: 0=Greedy, 1=Moderate, 2=Conservative
ADVERTISERS = [
    ("Nike", "Nike running shoes marathon athletic footwear trail running sneakers", 12.0, 4.00, 0.55, 8000, 0),
    ("Peloton", "Peloton home fitness spin bike connected workout cycling exercise", 10.0, 3.50, 0.50, 7000, 1),
    ("Gymshark", "Gymshark gym apparel workout clothing fitness activewear tank tops", 8.0, 2.80, 0.50, 5000, 0),
    ("Whoop", "Whoop fitness tracker recovery monitoring wearable health data band", 9.0, 3.00, 0.45, 6000, 2),
    ("Lululemon", "Lululemon yoga pants athleisure activewear women leggings sports bra", 11.0, 3.80, 0.55, 7500, 1),
    ("Zara", "Zara fast fashion trendy clothing seasonal outfits affordable style", 9.0, 3.20, 0.60, 6500, 0),
    ("Everlane", "Everlane sustainable basics minimalist wardrobe ethical fashion capsule", 7.0, 2.50, 0.45, 4000, 2),
    ("AthleticGreens", "Athletic Greens AG1 daily supplement greens superfood powder nutrition", 8.0, 3.00, 0.40, 5500, 1),
    ("Headspace", "Headspace meditation app mindfulness stress relief sleep calm focus", 7.0, 2.20, 0.45, 4500, 2),
    ("Noom", "Noom weight management healthy eating behavior change diet app coaching", 9.0, 3.20, 0.50, 6000, 1),
    ("RogueFitness", "Rogue Fitness powerlifting equipment barbell squat rack bumper plates", 10.0, 3.50, 0.40, 5000, 0),
    ("LaSportiva", "La Sportiva climbing shoes bouldering mountaineering rock climbing gear", 6.0, 1.80, 0.35, 3000, 2),
    ("PrecisionNutrition", "Precision Nutrition sports dietetics coaching macro meal planning athletes", 7.0, 2.40, 0.40, 4000, 1),
    ("AppleWatch", "Apple Watch smartwatch health fitness tracking heart rate ECG workout", 12.0, 4.50, 0.60, 9000, 1),
    ("Dyson", "Dyson home appliances air purifier vacuum cleaner hair dryer technology", 8.0, 2.80, 0.35, 5000, 2),
]

# --- Impression query clusters ---
# Each cluster: (name, weight, list of queries)
CLUSTERS = [
    ("running", 0.18, [
        "best running shoes for beginners",
        "marathon training plan 16 weeks",
        "how to improve 5k time",
        "running shoes for flat feet recommendations",
        "couch to half marathon training program",
        "best cardio exercises for weight loss",
        "treadmill vs outdoor running comparison",
        "GPS running watch heart rate monitor",
        "compression socks benefits for runners",
        "interval training speed workout plan",
    ]),
    ("yoga", 0.14, [
        "yoga for beginners at home routine",
        "best yoga mat thick cushion non slip",
        "morning stretching routine for flexibility",
        "yoga pants high waist comfortable leggings",
        "meditation and yoga retreat weekend",
        "hip opener yoga sequence for runners",
        "yoga blocks and props for beginners",
        "prenatal yoga exercises safe poses",
        "yoga teacher training certification online",
        "restorative yoga for stress and anxiety",
    ]),
    ("fashion", 0.22, [
        "summer outfit ideas women casual",
        "men's casual wear trends this season",
        "sustainable fashion brands affordable",
        "what to wear to a summer wedding",
        "capsule wardrobe essentials minimalist",
        "designer bags on sale outlet",
        "street style fashion inspiration lookbook",
        "professional workwear office outfits",
        "vintage clothing stores online thrift",
        "athleisure everyday outfits comfortable style",
    ]),
    ("strength", 0.15, [
        "home gym equipment essentials beginner",
        "beginner weight lifting program full body",
        "squat rack for small home gym",
        "best protein powder for muscle building",
        "deadlift form and technique guide",
        "resistance bands workout full body routine",
        "powerlifting competition preparation training",
        "adjustable dumbbells set review comparison",
        "muscle building meal plan high protein",
        "pre workout supplements best ingredients",
    ]),
    ("nutrition", 0.17, [
        "healthy meal prep ideas for the week",
        "weight loss meal plan calorie deficit",
        "macro counting for beginners guide",
        "vegan protein sources complete amino acids",
        "anti-inflammatory diet foods to eat",
        "intermittent fasting guide for beginners",
        "gut health supplements probiotics prebiotics",
        "low carb recipes easy weeknight dinner",
        "sports nutrition for endurance athletes fueling",
        "daily vitamins and supplements what to take",
    ]),
    ("wellness", 0.14, [
        "meditation app for anxiety and stress",
        "stress management techniques for work",
        "sleep hygiene tips for better rest",
        "mindfulness exercises for daily practice",
        "burnout recovery strategies self care",
        "therapy vs counseling what is the difference",
        "journaling prompts for mental health",
        "breathing exercises for calm and focus",
        "digital detox challenge one week plan",
        "work life balance tips remote workers",
    ]),
]


def fmt_vec(vec, per_line=8):
    """Format a float64 vector as Go literal lines."""
    lines = []
    for i in range(0, len(vec), per_line):
        chunk = vec[i:i+per_line]
        lines.append("\t\t" + ", ".join(f"{v:.6f}" for v in chunk) + ",")
    return "\n".join(lines)


def main():
    print("Loading BGE-small-en-v1.5 model...")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")

    # Embed advertiser descriptions
    adv_texts = [a[1] for a in ADVERTISERS]
    print(f"Embedding {len(adv_texts)} advertiser descriptions...")
    adv_embeddings = list(model.embed(adv_texts))

    # Embed all queries
    all_queries = []
    cluster_starts = []  # (cluster_idx, start, end)
    for ci, (name, weight, queries) in enumerate(CLUSTERS):
        start = len(all_queries)
        all_queries.extend(queries)
        cluster_starts.append((ci, start, len(all_queries)))

    print(f"Embedding {len(all_queries)} impression queries...")
    query_embeddings = list(model.embed(all_queries))

    dim = len(adv_embeddings[0])
    print(f"Embedding dimension: {dim}")

    # Compute some distance stats for calibration
    print("\n--- Distance stats (squared Euclidean, normalized embeddings) ---")
    for i, (name, *_) in enumerate(ADVERTISERS[:5]):
        for j, (name2, *_) in enumerate(ADVERTISERS[:5]):
            if i < j:
                d2 = np.sum((adv_embeddings[i] - adv_embeddings[j])**2)
                cos = np.dot(adv_embeddings[i], adv_embeddings[j])
                print(f"  {name} <-> {name2}: dist²={d2:.4f}  cos={cos:.4f}")

    # Write Go file
    out_path = "cmd/simulate/embeddings.go"
    print(f"\nWriting {out_path}...")
    with open(out_path, "w") as f:
        f.write("package main\n\n")
        f.write("// Code generated by gen_embeddings.py using BAAI/bge-small-en-v1.5 (384D). DO NOT EDIT.\n\n")
        f.write(f"const embeddingDim = {dim}\n\n")

        # Advertiser data struct
        f.write("type advData struct {\n")
        f.write("\tName      string\n")
        f.write("\tEmbedding []float64\n")
        f.write("\tMaxValue  float64\n")
        f.write("\tBaseBid   float64\n")
        f.write("\tBaseSigma float64\n")
        f.write("\tBaseBudget float64\n")
        f.write("\tStrategy  Strategy\n")
        f.write("}\n\n")

        # Advertiser embeddings
        f.write("var advertiserData = []advData{\n")
        for i, (name, desc, maxval, bid, sigma, budget, strat) in enumerate(ADVERTISERS):
            strat_name = ["Greedy", "Moderate", "Conservative"][strat]
            f.write(f'\t{{ // {name}: "{desc[:60]}..."\n')
            f.write(f'\t\tName: "{name}", MaxValue: {maxval}, BaseBid: {bid}, BaseSigma: {sigma}, BaseBudget: {budget}, Strategy: {strat_name},\n')
            f.write(f"\t\tEmbedding: []float64{{\n")
            f.write(fmt_vec(adv_embeddings[i]))
            f.write("\n\t\t},\n")
            f.write("\t},\n")
        f.write("}\n\n")

        # Query cluster struct
        f.write("type queryCluster struct {\n")
        f.write("\tName    string\n")
        f.write("\tWeight  float64\n")
        f.write("\tQueries [][]float64\n")
        f.write("}\n\n")

        # Query clusters
        f.write("var impressionClusters = []queryCluster{\n")
        for ci, (name, weight, queries) in enumerate(CLUSTERS):
            _, start, end = cluster_starts[ci]
            f.write(f'\t{{Name: "{name}", Weight: {weight}, Queries: [][]float64{{\n')
            for qi in range(start, end):
                query_text = all_queries[qi]
                f.write(f'\t\t{{ // "{query_text}"\n')
                f.write(fmt_vec(query_embeddings[qi]))
                f.write("\n\t\t},\n")
            f.write("\t}},\n")
        f.write("}\n")

    print(f"Done. Generated {out_path} ({dim}D, {len(ADVERTISERS)} advertisers, {len(all_queries)} queries)")


if __name__ == "__main__":
    main()
