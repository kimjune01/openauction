#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastembed"]
# ///
"""Generate real embeddings for the auction simulation (v3: near-miss niches).

Uses BAAI/bge-small-en-v1.5 (384D, open-weight) via fastembed (ONNX runtime).
Outputs embeddings.go with advertiser and query vectors as Go literals.

Usage:
    cd /Users/junekim/Documents/openauction
    uv run cmd/simulate/gen_embeddings.py
"""

import numpy as np
from fastembed import TextEmbedding

# --- Advertisers ---
# Each entry: (name, description for embedding, max_value, base_bid, base_sigma, base_budget)
# Cluster assignment is implicit by position (0=PT, 1=Fitness, 2=Nutrition, 3=Tutoring)
ADVERTISERS = [
    # Physical Therapy cluster (5) — keyword: "physical therapy"
    ("ClimbingPT", "physical therapy for rock climbers finger pulley A2 injury crimp rehab bouldering", 10.0, 3.5, 0.45, 6000),
    ("SportsPT", "sports physical therapy ACL recovery athletic injury return to play", 10.0, 3.5, 0.45, 6000),
    ("PelvicFloorPT", "pelvic floor physical therapy postpartum incontinence diastasis recti women's health", 9.0, 3.0, 0.45, 5500),
    ("PediatricPT", "pediatric physical therapy child motor development cerebral palsy early intervention", 8.0, 2.8, 0.45, 5000),
    ("GeneralPT", "physical therapy rehabilitation pain management back pain recovery", 8.0, 3.0, 0.50, 6000),

    # Fitness Coaching cluster (4) — keyword: "fitness coach"
    ("ClimbingCoach", "rock climbing coaching technique bouldering training movement skill beta", 9.0, 3.2, 0.45, 5500),
    ("RunningCoach", "marathon running coach 5k training plan race pace interval speed", 9.0, 3.2, 0.45, 5500),
    ("CrossFitCoach", "CrossFit coaching WOD functional fitness Olympic lifting competition prep", 9.0, 3.2, 0.45, 5500),
    ("PersonalTrainer", "personal trainer fitness workout strength training exercise coaching", 8.0, 3.0, 0.50, 5500),

    # Nutrition cluster (4) — keyword: "nutritionist"
    ("SportsDietitian", "sports dietitian endurance athlete fueling race day nutrition carb loading", 9.0, 3.0, 0.45, 5500),
    ("GutHealth", "gut health nutritionist SIBO IBS microbiome digestive wellness elimination diet", 8.0, 2.8, 0.45, 5000),
    ("WeightLossCoach", "weight loss nutritionist calorie deficit macro counting portion control meal plan", 9.0, 3.0, 0.45, 5500),
    ("GeneralNutritionist", "registered dietitian nutrition counseling healthy eating balanced diet meal planning", 7.0, 2.5, 0.50, 5000),

    # Tutoring cluster (2) — keyword: "math tutor"
    ("ADHDMathTutor", "math tutoring for ADHD students hands-on learning executive function support", 8.0, 2.8, 0.45, 4500),
    ("GeneralMathTutor", "math tutoring algebra calculus SAT prep test preparation homework help", 7.0, 2.5, 0.50, 4500),
]

# Cluster boundaries: PT=0..4, Fitness=5..8, Nutrition=9..12, Tutoring=13..14
CLUSTER_BOUNDARIES = [
    (0, 5),   # PT
    (5, 9),   # Fitness
    (9, 13),  # Nutrition
    (13, 15), # Tutoring
]

# --- Impression query clusters ---
# Each cluster: (name, weight, list of queries)
CLUSTERS = [
    ("physical_therapy", 0.35, [
        # Specialist (4)
        "finger pulley injury from rock climbing crimping",
        "A2 pulley rehab protocol for bouldering",
        "pelvic floor exercises after C-section delivery",
        "potty training regression toddler physical therapy",
        # Boundary (4)
        "shoulder injury from overhead sport",
        "hip flexor tightness from running and climbing",
        "core stability exercises postpartum return to sport",
        "growing pains in active child athlete",
        # General (4)
        "physical therapy for lower back pain",
        "how to find a good physical therapist near me",
        "physical therapy vs chiropractor for pain",
        "does physical therapy actually work",
    ]),
    ("fitness_coaching", 0.25, [
        # Specialist (3)
        "how to train finger strength for climbing V7",
        "16 week marathon training plan sub 3 hours",
        "CrossFit open workout strategy tips",
        # Boundary (3)
        "strength training for endurance athletes",
        "grip strength training for athletes",
        "HIIT vs steady state cardio for fat loss",
        # General (4)
        "how to get in shape as a beginner",
        "best exercise routine for weight loss",
        "finding a good fitness coach online",
        "workout plan for busy professionals",
    ]),
    ("nutrition", 0.25, [
        # Specialist (3)
        "what to eat before a marathon race day",
        "low FODMAP diet for IBS symptom relief",
        "macro split for cutting weight lifting",
        # Boundary (3)
        "protein timing around workouts for muscle",
        "bloating after high protein diet",
        "meal prep for athletes on a budget",
        # General (4)
        "healthy eating tips for beginners",
        "how to eat better without dieting",
        "should I see a nutritionist or dietitian",
        "balanced meal plan for the week",
    ]),
    ("tutoring", 0.15, [
        # Specialist (2)
        "math tutor for child with ADHD attention issues",
        "SAT math prep tutoring intensive course",
        # Boundary (2)
        "my kid struggles with math motivation focus",
        "hands-on math activities for kids who hate worksheets",
        # General (2)
        "find a math tutor near me",
        "online math tutoring for middle school",
    ]),
]

# Query type metadata: specialist=0, boundary=1, general=2
QUERY_TYPES = {
    "physical_therapy": [0]*4 + [1]*4 + [2]*4,
    "fitness_coaching": [0]*3 + [1]*3 + [2]*4,
    "nutrition":        [0]*3 + [1]*3 + [2]*4,
    "tutoring":         [0]*2 + [1]*2 + [2]*2,
}


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
    all_types = []
    cluster_starts = []  # (cluster_idx, start, end)
    for ci, (name, weight, queries) in enumerate(CLUSTERS):
        start = len(all_queries)
        all_queries.extend(queries)
        all_types.extend(QUERY_TYPES[name])
        cluster_starts.append((ci, start, len(all_queries)))

    print(f"Embedding {len(all_queries)} impression queries...")
    query_embeddings = list(model.embed(all_queries))

    dim = len(adv_embeddings[0])
    print(f"Embedding dimension: {dim}")

    # Compute some distance stats for calibration
    print("\n--- Intra-cluster distance stats (squared Euclidean, normalized embeddings) ---")
    for ci, (cstart, cend) in enumerate(CLUSTER_BOUNDARIES):
        cluster_name = ["PT", "Fitness", "Nutrition", "Tutoring"][ci]
        for i in range(cstart, cend):
            for j in range(i+1, cend):
                d2 = np.sum((adv_embeddings[i] - adv_embeddings[j])**2)
                cos = np.dot(adv_embeddings[i], adv_embeddings[j])
                print(f"  [{cluster_name}] {ADVERTISERS[i][0]} <-> {ADVERTISERS[j][0]}: dist²={d2:.4f}  cos={cos:.4f}")

    # Write Go file
    out_path = "cmd/simulate/embeddings.go"
    print(f"\nWriting {out_path}...")
    with open(out_path, "w") as f:
        f.write("package main\n\n")
        f.write("// Code generated by gen_embeddings.py using BAAI/bge-small-en-v1.5 (384D). DO NOT EDIT.\n\n")
        f.write(f"const embeddingDim = {dim}\n\n")

        # Query type enum
        f.write("// Query type classification\n")
        f.write("const (\n")
        f.write("\tQuerySpecialist = 0\n")
        f.write("\tQueryBoundary   = 1\n")
        f.write("\tQueryGeneral    = 2\n")
        f.write(")\n\n")

        # Advertiser data struct
        f.write("type advData struct {\n")
        f.write("\tName       string\n")
        f.write("\tEmbedding  []float64\n")
        f.write("\tMaxValue   float64\n")
        f.write("\tBaseBid    float64\n")
        f.write("\tBaseSigma  float64\n")
        f.write("\tBaseBudget float64\n")
        f.write("\tCluster    int\n")
        f.write("}\n\n")

        # Advertiser embeddings
        f.write("var advertiserData = []advData{\n")
        for i, (name, desc, maxval, bid, sigma, budget) in enumerate(ADVERTISERS):
            # Determine cluster
            cluster = 0
            for ci, (cstart, cend) in enumerate(CLUSTER_BOUNDARIES):
                if cstart <= i < cend:
                    cluster = ci
                    break
            f.write(f'\t{{ // {name}: "{desc[:60]}..."\n')
            f.write(f'\t\tName: "{name}", MaxValue: {maxval}, BaseBid: {bid}, BaseSigma: {sigma}, BaseBudget: {budget}, Cluster: {cluster},\n')
            f.write(f"\t\tEmbedding: []float64{{\n")
            f.write(fmt_vec(adv_embeddings[i]))
            f.write("\n\t\t},\n")
            f.write("\t},\n")
        f.write("}\n\n")

        # Query cluster struct (with Types)
        f.write("type queryCluster struct {\n")
        f.write("\tName    string\n")
        f.write("\tWeight  float64\n")
        f.write("\tQueries [][]float64\n")
        f.write("\tTypes   []int\n")
        f.write("}\n\n")

        # Query clusters
        f.write("var impressionClusters = []queryCluster{\n")
        for ci, (name, weight, queries) in enumerate(CLUSTERS):
            _, start, end = cluster_starts[ci]
            types_slice = all_types[start:end]
            types_str = ", ".join(str(t) for t in types_slice)
            f.write(f'\t{{Name: "{name}", Weight: {weight}, Types: []int{{{types_str}}}, Queries: [][]float64{{\n')
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
