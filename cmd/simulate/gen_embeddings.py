#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastembed"]
# ///
"""Generate embeddings for auction simulation v3.5 (Hotelling drift × cluster tightness).

5 clusters with controlled tightness levels:
- Tight (Physical Therapy, target cos > 0.90): Nearly identical descriptions
- Medium (Fitness Coaching, target cos ~0.80): Shared base, moderate niche differentiation
- Loose (Wellness Professionals, target cos ~0.65-0.70): Same broad market, very different specialties
- Very Tight (Yoga instructors, target cos > 0.95): Minimal aspect swap
- Very Loose (Unrelated trades, target cos ~0.50): Completely different industries

Uses BAAI/bge-small-en-v1.5 (384D) via fastembed.
"""

import numpy as np
from fastembed import TextEmbedding

# --- Advertisers ---
# Each entry: (name, description, max_value, base_bid, base_sigma, base_budget, is_specialist, cluster_label)
# cluster_label: "tight", "medium", "loose", "very_tight", "very_loose" — used to assign cluster indices
ADVERTISERS = [
    # --- Tight cluster (0) — Physical Therapy (target cos > 0.90) ---
    # Identical long base, single-word swap at end for maximum similarity
    ("KneePT",
     "licensed physical therapist providing rehabilitation therapy exercise recovery treatment for patients with knee injury",
     12.0, 3.5, 0.30, 6000, True, "tight"),
    ("ShoulderPT",
     "licensed physical therapist providing rehabilitation therapy exercise recovery treatment for patients with shoulder injury",
     12.0, 3.5, 0.30, 6000, True, "tight"),
    ("BackPT",
     "licensed physical therapist providing rehabilitation therapy exercise recovery treatment for patients with back injury",
     12.0, 3.0, 0.30, 5500, True, "tight"),
    ("HipPT",
     "licensed physical therapist providing rehabilitation therapy exercise recovery treatment for patients with hip injury",
     12.0, 2.8, 0.30, 5000, True, "tight"),
    ("NeckPT",
     "licensed physical therapist providing rehabilitation therapy exercise recovery treatment for patients with neck injury",
     6.0, 3.0, 0.55, 6000, False, "tight"),

    # --- Medium cluster (1) — Fitness Coaching (target cos ~0.80) ---
    # Shared role, moderate niche language
    ("ClimbingCoach",
     "certified fitness coach providing training programs specializing in rock climbing bouldering technique",
     12.0, 3.2, 0.30, 5500, True, "medium"),
    ("MarathonCoach",
     "certified fitness coach providing training programs specializing in marathon running race preparation",
     12.0, 3.2, 0.30, 5500, True, "medium"),
    ("CrossFitCoach",
     "certified fitness coach providing training programs specializing in CrossFit functional fitness",
     12.0, 3.2, 0.30, 5500, True, "medium"),
    ("YogaCoach",
     "certified fitness coach providing training programs specializing in yoga flexibility mindfulness",
     12.0, 3.0, 0.30, 5500, True, "medium"),
    ("PersonalTrainer",
     "certified fitness coach providing training programs specializing in general personal strength",
     6.0, 3.0, 0.55, 5500, False, "medium"),

    # --- Loose cluster (2) — Wellness Professionals (target cos ~0.65-0.70) ---
    # Same broad market, very different specialties
    ("SportsDietitian",
     "registered dietitian sports nutrition endurance athlete fueling",
     12.0, 3.0, 0.30, 5500, True, "loose"),
    ("CBTTherapist",
     "licensed therapist cognitive behavioral anxiety depression mental health",
     12.0, 2.8, 0.30, 5000, True, "loose"),
    ("MassageTherapist",
     "certified massage therapist deep tissue sports recovery",
     12.0, 3.0, 0.30, 5500, True, "loose"),
    ("Acupuncturist",
     "acupuncturist traditional Chinese medicine pain chronic conditions",
     12.0, 2.8, 0.30, 5000, True, "loose"),
    ("HealthCoach",
     "health coach lifestyle wellness stress sleep habit coaching",
     6.0, 2.5, 0.55, 5000, False, "loose"),

    # --- Very Tight cluster (3) — Yoga studios (target cos > 0.95) ---
    # Duplicate the exact same description to guarantee near-1.0 cosine
    # All five are the same yoga studio ad text — only the Go-side Name differs
    ("YogaStudioA",
     "certified yoga instructor offering group classes private sessions beginner intermediate advanced vinyasa hatha restorative yoga downtown studio",
     12.0, 3.5, 0.30, 6000, True, "very_tight"),
    ("YogaStudioB",
     "certified yoga instructor offering group classes private sessions beginner intermediate advanced vinyasa hatha restorative yoga midtown studio",
     12.0, 3.5, 0.30, 6000, True, "very_tight"),
    ("YogaStudioC",
     "certified yoga instructor offering group classes private sessions beginner intermediate advanced vinyasa hatha restorative yoga uptown studio",
     12.0, 3.0, 0.30, 5500, True, "very_tight"),
    ("YogaStudioD",
     "certified yoga instructor offering group classes private sessions beginner intermediate advanced vinyasa hatha restorative yoga eastside studio",
     12.0, 2.8, 0.30, 5000, True, "very_tight"),
    ("YogaStudioE",
     "certified yoga instructor offering group classes private sessions beginner intermediate advanced vinyasa hatha restorative yoga westside studio",
     6.0, 3.0, 0.55, 6000, False, "very_tight"),

    # --- Very Loose cluster (4) — Unrelated trades (target cos ~0.50) ---
    # Completely different industries, minimal semantic overlap
    ("AutoMechanic",
     "automotive mechanic brake repair engine diagnostic transmission service",
     12.0, 3.0, 0.30, 5500, True, "very_loose"),
    ("RealEstateAgent",
     "real estate agent home buying property listing residential sales",
     12.0, 3.2, 0.30, 5500, True, "very_loose"),
    ("WeddingPhotographer",
     "wedding photographer event photography portrait session engagement",
     12.0, 3.0, 0.30, 5500, True, "very_loose"),
    ("Plumber",
     "licensed plumber pipe repair leak detection water heater installation",
     12.0, 2.8, 0.30, 5000, True, "very_loose"),
    ("Electrician",
     "certified electrician wiring installation electrical panel upgrade",
     6.0, 2.5, 0.55, 5000, False, "very_loose"),
]

CLUSTER_LABELS = ["tight", "medium", "loose", "very_tight", "very_loose"]

# --- Impression query clusters ---
# ~10-12 queries per cluster across specialist/boundary/general tiers
CLUSTERS = [
    ("tight_pt", 0.20, [
        # Specialist queries (clearly map to one PT niche)
        "knee injury physical therapy rehabilitation",
        "shoulder injury physical therapy recovery",
        "back pain physical therapy treatment exercises",
        "hip injury rehabilitation physical therapy",
        # Boundary queries (between PT niches)
        "joint injury rehabilitation therapy exercise",
        "upper body injury physical therapy recovery",
        "lower body injury rehabilitation treatment",
        "physical therapy injury recovery exercise plan",
        # General queries
        "physical therapy for pain relief treatment",
        "how to find a good physical therapist",
        "does physical therapy actually work",
    ]),
    ("medium_fitness", 0.20, [
        # Specialist queries
        "rock climbing training plan bouldering technique",
        "marathon training program race preparation running",
        "CrossFit workout functional fitness programming",
        "yoga flexibility mindfulness practice routine",
        # Boundary queries
        "strength training for endurance athletes",
        "grip strength training for sport performance",
        "HIIT cardio training program athletic",
        "flexibility mobility training for athletes",
        # General queries
        "how to get in shape exercise routine",
        "best workout plan for beginners",
        "finding a good fitness coach online",
    ]),
    ("loose_wellness", 0.20, [
        # Specialist queries
        "sports nutrition meal plan endurance athlete",
        "cognitive behavioral therapy anxiety treatment",
        "deep tissue massage sports recovery session",
        "acupuncture chronic pain treatment traditional",
        # Boundary queries
        "stress management holistic wellness approach",
        "recovery methods for athletic performance",
        "natural pain relief alternative medicine options",
        "mental health wellness lifestyle improvement",
        # General queries
        "health and wellness professional near me",
        "how to improve overall wellbeing lifestyle",
        "holistic health practitioner consultation",
    ]),
    ("very_tight_yoga", 0.20, [
        # Specialist queries
        "yoga class flexibility stretching routine",
        "yoga class strength building poses",
        "yoga class balance coordination practice",
        "yoga class relaxation meditation breathing",
        # Boundary queries
        "group yoga class for beginners all levels",
        "yoga instructor near me weekly classes",
        "certified yoga teacher private session",
        "yoga class morning routine daily practice",
        # General queries
        "best yoga classes in my area",
        "how to start practicing yoga",
        "yoga instructor certification training",
    ]),
    ("very_loose_trades", 0.20, [
        # Specialist queries
        "car brake repair mechanic near me",
        "home buying real estate agent listing",
        "wedding photographer booking engagement photos",
        "plumber pipe leak repair emergency",
        "electrician wiring panel installation",
        # Boundary queries
        "home repair service professional contractor",
        "local service provider appointment booking",
        "professional trade services estimate quote",
        # General queries
        "find a local professional service provider",
        "best rated service professionals near me",
        "how to hire a reliable contractor",
    ]),
]


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

    # Assign cluster indices from labels (0=tight, 1=medium, 2=loose)
    cluster_map = {label: idx for idx, label in enumerate(CLUSTER_LABELS)}
    labels = [cluster_map[a[7]] for a in ADVERTISERS]

    # Print clusters
    cluster_members = {}
    for i, label in enumerate(labels):
        cluster_members.setdefault(label, []).append(ADVERTISERS[i][0])

    print("\n  Cluster assignments:")
    for label in sorted(cluster_members.keys()):
        members = cluster_members[label]
        tightness = CLUSTER_LABELS[label]
        print(f"    Cluster {label} ({tightness}): {', '.join(members)}")

    # Print intra-cluster cosine similarities
    print("\n  Intra-cluster cosine similarities:")
    for cl in range(len(CLUSTER_LABELS)):
        cos_vals = []
        cl_indices = [i for i, l in enumerate(labels) if l == cl]
        for ii in range(len(cl_indices)):
            for jj in range(ii+1, len(cl_indices)):
                i, j = cl_indices[ii], cl_indices[jj]
                cos = float(np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j])))
                cos_vals.append(cos)
                print(f"    [c{cl} {CLUSTER_LABELS[cl]}] {ADVERTISERS[i][0]} <-> {ADVERTISERS[j][0]}: cos={cos:.4f}")
        if cos_vals:
            print(f"    → Cluster {cl} ({CLUSTER_LABELS[cl]}) mean cos: {np.mean(cos_vals):.4f}, min: {np.min(cos_vals):.4f}, max: {np.max(cos_vals):.4f}")
        print()

    # Print cross-cluster distances
    print("  Cross-cluster nearest pairs:")
    for ci in range(len(CLUSTER_LABELS)):
        for cj in range(ci+1, len(CLUSTER_LABELS)):
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

        # Query type enum
        f.write("// Query type classification (assigned at runtime by computeQueryTypes)\n")
        f.write("const (\n")
        f.write("\tQuerySpecialist = 0\n")
        f.write("\tQueryBoundary   = 1\n")
        f.write("\tQueryGeneral    = 2\n")
        f.write(")\n\n")

        # Cluster tightness labels
        labels_str = ", ".join(f'"{l}"' for l in CLUSTER_LABELS)
        f.write(f"// Cluster tightness labels\n")
        f.write(f"var clusterTightnessLabels = []string{{{labels_str}}}\n\n")

        # Advertiser data struct
        f.write("type advData struct {\n")
        f.write("\tName         string\n")
        f.write("\tEmbedding    []float64\n")
        f.write("\tMaxValue     float64\n")
        f.write("\tBaseBid      float64\n")
        f.write("\tBaseSigma    float64\n")
        f.write("\tBaseBudget   float64\n")
        f.write("\tCluster      int\n")
        f.write("\tIsSpecialist bool\n")
        f.write("}\n\n")

        # Advertiser embeddings
        f.write("var advertiserData = []advData{\n")
        for i, adv in enumerate(ADVERTISERS):
            name, desc, maxval, bid, sigma, budget, is_spec, _ = adv
            spec_str = "true" if is_spec else "false"
            f.write(f'\t{{ // {name}: "{desc[:60]}..."\n')
            f.write(f'\t\tName: "{name}", MaxValue: {maxval}, BaseBid: {bid}, BaseSigma: {sigma}, BaseBudget: {budget}, Cluster: {labels[i]}, IsSpecialist: {spec_str},\n')
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
        f.write("\tLabels  []string\n")
        f.write("}\n\n")

        # Query clusters
        f.write("var impressionClusters = []queryCluster{\n")
        for ci, (name, weight, queries) in enumerate(CLUSTERS):
            _, start, end = cluster_starts[ci]
            labels_go = ", ".join(f'"{make_label(q)}"' for q in queries)
            f.write(f'\t{{Name: "{name}", Weight: {weight}, Labels: []string{{{labels_go}}}, Queries: [][]float64{{\n')
            for qi in range(start, end):
                query_text = all_queries[qi]
                f.write(f'\t\t{{ // "{query_text}"\n')
                f.write(fmt_vec(query_embeddings[qi]))
                f.write("\n\t\t},\n")
            f.write("\t}},\n")
        f.write("}\n")

    print(f"Done. Generated {out_path} ({dim}D, {len(ADVERTISERS)} advertisers, {len(all_queries)} queries, {len(CLUSTER_LABELS)} tightness clusters)")


if __name__ == "__main__":
    main()
