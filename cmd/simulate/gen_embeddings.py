#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastembed"]
# ///
"""Generate embeddings for auction simulation v3.3 (tighter specialist clustering).

Key changes from v3.2:
- Rewritten advertiser descriptions share heavy base phrases within each cluster
- Niche is a modifier, not a replacement — targets intra-cluster cosine > 0.80
- Makes keyword bins honestly coarse (close specialists land in same bin naturally)
- Makes embedding separation harder (real work to distinguish near-miss specialists)

Uses BAAI/bge-small-en-v1.5 (384D) via fastembed.
"""

import numpy as np
from fastembed import TextEmbedding

# --- Advertisers ---
# Each entry: (name, description, max_value, base_bid, base_sigma, base_budget)
ADVERTISERS = [
    # Physical Therapy variants (5) — heavy shared base, niche modifier at end
    ("ClimbingPT", "licensed physical therapist providing rehabilitation exercise therapy injury recovery specializing in rock climbing finger pulley", 10.0, 3.5, 0.45, 6000),
    ("SportsPT", "licensed physical therapist providing rehabilitation exercise therapy injury recovery specializing in sports ACL athletic", 10.0, 3.5, 0.45, 6000),
    ("PelvicFloorPT", "licensed physical therapist providing rehabilitation exercise therapy injury recovery specializing in pelvic floor postpartum", 9.0, 3.0, 0.45, 5500),
    ("PediatricPT", "licensed physical therapist providing rehabilitation exercise therapy injury recovery specializing in pediatric child motor", 8.0, 2.8, 0.45, 5000),
    ("GeneralPT", "licensed physical therapist providing rehabilitation exercise therapy injury recovery specializing in back pain general wellness", 8.0, 3.0, 0.50, 6000),

    # Fitness Coaching variants (4) — heavy shared base, niche modifier at end
    ("ClimbingCoach", "certified fitness coach offering training workout programs and coaching specializing in rock climbing bouldering", 9.0, 3.2, 0.45, 5500),
    ("RunningCoach", "certified fitness coach offering training workout programs and coaching specializing in marathon running race prep", 9.0, 3.2, 0.45, 5500),
    ("CrossFitCoach", "certified fitness coach offering training workout programs and coaching specializing in CrossFit functional fitness", 9.0, 3.2, 0.45, 5500),
    ("PersonalTrainer", "certified fitness coach offering training workout programs and coaching specializing in personal strength training", 8.0, 3.0, 0.50, 5500),

    # Nutrition variants (4) — heavy shared base, niche modifier at end
    ("SportsDietitian", "registered dietitian providing nutrition counseling meal planning and diet guidance specializing in sports athletes endurance", 9.0, 3.0, 0.45, 5500),
    ("GutHealth", "registered dietitian providing nutrition counseling meal planning and diet guidance specializing in gut health IBS digestive", 8.0, 2.8, 0.45, 5000),
    ("WeightLossCoach", "registered dietitian providing nutrition counseling meal planning and diet guidance specializing in weight loss calorie deficit", 9.0, 3.0, 0.45, 5500),
    ("GeneralNutritionist", "registered dietitian providing nutrition counseling meal planning and diet guidance specializing in healthy eating balanced", 7.0, 2.5, 0.50, 5000),

    # Tutoring variants (2) — heavy shared base, niche modifier at end
    ("ADHDMathTutor", "experienced math tutor providing tutoring lessons homework help and academic support for students with ADHD attention focus", 8.0, 2.8, 0.45, 4500),
    ("GeneralMathTutor", "experienced math tutor providing tutoring lessons homework help and academic support for students SAT test prep algebra calculus", 7.0, 2.5, 0.50, 4500),
]

# --- Impression query clusters (expanded: 62 queries) ---
CLUSTERS = [
    ("physical_therapy", 0.35, [
        "finger pulley injury from rock climbing crimping",
        "A2 pulley rehab protocol for bouldering",
        "pelvic floor exercises after C-section delivery",
        "potty training regression toddler physical therapy",
        "ACL rehab exercises after knee surgery recovery",
        "postpartum diastasis recti recovery therapy",
        "shoulder injury from overhead sport",
        "hip flexor tightness from running and climbing",
        "core stability exercises postpartum return to sport",
        "growing pains in active child athlete",
        "sports injury rehabilitation return to play protocol",
        "pregnancy exercise safe physical therapy guidance",
        "physical therapy for lower back pain",
        "how to find a good physical therapist near me",
        "physical therapy vs chiropractor for pain",
        "does physical therapy actually work",
        "physical therapy exercises I can do at home",
        "how long does physical therapy take to work",
    ]),
    ("fitness_coaching", 0.25, [
        "how to train finger strength for climbing V7",
        "16 week marathon training plan sub 3 hours",
        "CrossFit open workout strategy tips",
        "rock climbing training plan intermediate boulderer",
        "beginner running program couch to 5k",
        "Olympic weightlifting snatch technique coaching",
        "strength training for endurance athletes",
        "grip strength training for athletes",
        "HIIT vs steady state cardio for fat loss",
        "strength and conditioning program for sport",
        "bodyweight workout plan no gym equipment",
        "how to get in shape as a beginner",
        "best exercise routine for weight loss",
        "finding a good fitness coach online",
        "workout plan for busy professionals",
        "how often should I exercise per week",
    ]),
    ("nutrition", 0.25, [
        "what to eat before a marathon race day",
        "low FODMAP diet for IBS symptom relief",
        "macro split for cutting weight lifting",
        "keto diet meal plan for weight loss beginners",
        "anti-inflammatory foods for gut healing",
        "sports nutrition supplements for endurance running",
        "protein timing around workouts for muscle",
        "bloating after high protein diet",
        "meal prep for athletes on a budget",
        "post-workout recovery shake protein smoothie recipe",
        "food sensitivity elimination diet protocol",
        "healthy eating tips for beginners",
        "how to eat better without dieting",
        "should I see a nutritionist or dietitian",
        "balanced meal plan for the week",
        "how many calories should I eat per day",
    ]),
    ("tutoring", 0.15, [
        "math tutor for child with ADHD attention issues",
        "SAT math prep tutoring intensive course",
        "AP calculus tutoring test prep advanced math",
        "learning disabilities math dyscalculia support tutoring",
        "my kid struggles with math motivation focus",
        "hands-on math activities for kids who hate worksheets",
        "fun math games for children who struggle",
        "how to help child with math anxiety frustration",
        "find a math tutor near me",
        "online math tutoring for middle school",
        "math tutoring rates per hour cost comparison",
        "after school math help program for students",
    ]),
]


def kmeans(X, k=4, max_iter=100, seed=42):
    """Simple k-means clustering on numpy arrays."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, k, replace=False)
    centroids = X[idx].copy()
    for _ in range(max_iter):
        dists = np.array([[np.sum((x - c)**2) for c in centroids] for x in X])
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids
    return labels.tolist(), centroids


def make_label(text, max_len=30):
    """Generate a short label from query text."""
    if len(text) <= max_len:
        return text
    words = text.split()
    label = ""
    for w in words:
        if label and len(label) + len(w) + 1 > max_len:
            break
        if label:
            label += " "
        label += w
    return label or text[:max_len]


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
    X = np.array(adv_embeddings)

    # K-means clustering
    print(f"\nRunning k-means (k=4) on {len(X)} advertiser embeddings...")
    labels, centroids = kmeans(X, k=4)

    # Print clusters
    cluster_members = {}
    for i, label in enumerate(labels):
        cluster_members.setdefault(label, []).append(ADVERTISERS[i][0])

    print("\n  K-means clusters:")
    for label in sorted(cluster_members.keys()):
        members = cluster_members[label]
        print(f"    Cluster {label}: {', '.join(members)}")

    # Print intra-cluster distances
    print("\n  Intra-cluster distances:")
    for i in range(len(ADVERTISERS)):
        for j in range(i+1, len(ADVERTISERS)):
            if labels[i] == labels[j]:
                cos = float(np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j])))
                print(f"    [c{labels[i]}] {ADVERTISERS[i][0]} <-> {ADVERTISERS[j][0]}: cos={cos:.4f}")

    # Print inter-cluster distances (nearest cross-cluster pairs)
    print("\n  Cross-cluster distances (nearest pair per cluster pair):")
    unique_labels = sorted(set(labels))
    for ci_idx in range(len(unique_labels)):
        for cj_idx in range(ci_idx+1, len(unique_labels)):
            ci, cj = unique_labels[ci_idx], unique_labels[cj_idx]
            best_cos = -1
            best_pair = ("", "")
            for i in range(len(ADVERTISERS)):
                for j in range(len(ADVERTISERS)):
                    if labels[i] == ci and labels[j] == cj:
                        cos = float(np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j])))
                        if cos > best_cos:
                            best_cos = cos
                            best_pair = (ADVERTISERS[i][0], ADVERTISERS[j][0])
            print(f"    c{ci}↔c{cj}: {best_pair[0]} <-> {best_pair[1]}: cos={best_cos:.4f}")

    # Embed all queries
    all_queries = []
    cluster_starts = []
    for ci, (name, weight, queries) in enumerate(CLUSTERS):
        start = len(all_queries)
        all_queries.extend(queries)
        cluster_starts.append((ci, start, len(all_queries)))

    print(f"\nEmbedding {len(all_queries)} impression queries...")
    query_embeddings = list(model.embed(all_queries))

    dim = len(adv_embeddings[0])
    print(f"Embedding dimension: {dim}")

    # Write Go file
    out_path = "cmd/simulate/embeddings.go"
    print(f"\nWriting {out_path}...")
    with open(out_path, "w") as f:
        f.write("package main\n\n")
        f.write("// Code generated by gen_embeddings.py using BAAI/bge-small-en-v1.5 (384D). DO NOT EDIT.\n\n")
        f.write(f"const embeddingDim = {dim}\n\n")

        # Query type enum (values assigned at runtime by Go)
        f.write("// Query type classification (assigned at runtime by computeQueryTypes)\n")
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
            f.write(f'\t{{ // {name}: "{desc[:60]}..."\n')
            f.write(f'\t\tName: "{name}", MaxValue: {maxval}, BaseBid: {bid}, BaseSigma: {sigma}, BaseBudget: {budget}, Cluster: {labels[i]},\n')
            f.write(f"\t\tEmbedding: []float64{{\n")
            f.write(fmt_vec(adv_embeddings[i]))
            f.write("\n\t\t},\n")
            f.write("\t},\n")
        f.write("}\n\n")

        # Query cluster struct (Labels for display, no Types — computed at runtime)
        f.write("type queryCluster struct {\n")
        f.write("\tName    string\n")
        f.write("\tWeight  float64\n")
        f.write("\tQueries [][]float64\n")
        f.write("\tLabels  []string\n")
        f.write("}\n\n")

        # Query clusters
        f.write("var impressionClusters = []queryCluster{\n")
        for ci, (name, weight, queries) in enumerate(CLUSTERS):
            _, start, end = cluster_starts[ci]
            # Labels
            labels_go = ", ".join(f'"{make_label(q)}"' for q in queries)
            f.write(f'\t{{Name: "{name}", Weight: {weight}, Labels: []string{{{labels_go}}}, Queries: [][]float64{{\n')
            for qi in range(start, end):
                query_text = all_queries[qi]
                f.write(f'\t\t{{ // "{query_text}"\n')
                f.write(fmt_vec(query_embeddings[qi]))
                f.write("\n\t\t},\n")
            f.write("\t}},\n")
        f.write("}\n")

    print(f"Done. Generated {out_path} ({dim}D, {len(ADVERTISERS)} advertisers, {len(all_queries)} queries, k-means clusters)")


if __name__ == "__main__":
    main()
