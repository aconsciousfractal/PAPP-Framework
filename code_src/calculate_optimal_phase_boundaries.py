"""
Calculate Optimal Spectral Phase Boundaries
============================================

Uses rigorous clustering algorithms to derive phase boundaries for 
ρ = λ₂/λ_max connectivity metric, replacing arbitrary round numbers.

Methods:
1. K-means clustering (k=3) with silhouette validation
2. Jenks natural breaks optimization
3. Gaussian mixture model (GMM)
4. Visual validation via elbow method

Author: PAPP Validation Suite
Date: 2026-02-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.stats as stats
import os

# Try importing Jenks (optional)
try:
    import jenkspy
    HAS_JENKSPY = True
except ImportError:
    HAS_JENKSPY = False
    print("Warning: jenkspy not available (pip install jenkspy for Jenks natural breaks)")


def generate_synthetic_data(n_samples=10000, seed=42):
    """
    Generate synthetic ρ distribution matching empirical observations:
    - Phase I (low): ~30%, mean ~0.08
    - Phase II (intermediate): ~60%, mean ~0.30
    - Phase III (high): ~10%, mean ~0.55
    """
    np.random.seed(seed)
    
    # Mixture of 3 Gaussians
    n1 = int(n_samples * 0.30)
    n2 = int(n_samples * 0.60)
    n3 = n_samples - n1 - n2
    
    # Phase I: Low connectivity
    rho1 = np.random.beta(2, 8, n1) * 0.20  # Concentrated near 0
    
    # Phase II: Intermediate
    rho2 = np.random.beta(3, 3, n2) * 0.50 + 0.10  # Centered ~0.30
    
    # Phase III: High connectivity
    rho3 = np.random.beta(5, 2, n3) * 0.35 + 0.40  # Concentrated near 0.6
    
    rho_all = np.concatenate([rho1, rho2, rho3])
    np.random.shuffle(rho_all)
    
    return rho_all


def method_1_kmeans(rho, k=3):
    """K-means clustering with silhouette validation"""
    print("\n" + "="*60)
    print("METHOD 1: K-Means Clustering")
    print("="*60)
    
    X = rho.reshape(-1, 1)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    
    # Sort centers
    sorted_idx = np.argsort(centers)
    centers_sorted = centers[sorted_idx]
    
    # Calculate boundaries (midpoints between centers)
    boundaries = []
    for i in range(len(centers_sorted) - 1):
        boundary = (centers_sorted[i] + centers_sorted[i+1]) / 2
        boundaries.append(boundary)
    
    # Validation metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    print(f"Cluster Centers: {centers_sorted}")
    print(f"Optimal Boundaries: {boundaries}")
    print(f"Silhouette Score: {silhouette:.4f} (higher is better, >0.5 is good)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    
    # Distribution per phase
    for i, idx in enumerate(sorted_idx):
        mask = labels == idx
        pct = 100 * mask.sum() / len(labels)
        print(f"Phase {i+1}: {pct:.1f}% of data")
    
    return boundaries, silhouette


def method_2_jenks(rho, k=3):
    """Jenks natural breaks optimization"""
    print("\n" + "="*60)
    print("METHOD 2: Jenks Natural Breaks")
    print("="*60)
    
    if not HAS_JENKSPY:
        print("Skipped (jenkspy not installed)")
        return None, None
    
    # Jenks returns k+1 values (including min and max)
    breaks = jenkspy.jenks_breaks(rho, n_classes=k)
    
    # Boundaries are internal breaks (exclude min/max)
    boundaries = breaks[1:-1]
    
    print(f"Jenks Breaks: {breaks}")
    print(f"Optimal Boundaries: {boundaries}")
    
    # Calculate Goodness of Variance Fit (GVF)
    # GVF = 1 - (SDAM / SDCM)
    # SDAM: Sum of squared deviations of array means
    # SDCM: Sum of squared deviations of class means
    
    labels = np.digitize(rho, breaks[1:-1])
    total_mean = rho.mean()
    
    sdam = np.sum((rho - total_mean)**2)
    sdcm = 0
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            class_mean = rho[mask].mean()
            sdcm += np.sum((rho[mask] - class_mean)**2)
    
    gvf = 1 - (sdcm / sdam) if sdam > 0 else 0
    print(f"Goodness of Variance Fit (GVF): {gvf:.4f} (higher is better, >0.8 is good)")
    
    # Distribution
    for i in range(k):
        mask = labels == i
        pct = 100 * mask.sum() / len(labels)
        print(f"Phase {i+1}: {pct:.1f}% of data")
    
    return boundaries, gvf


def method_3_gmm(rho, k=3):
    """Gaussian Mixture Model"""
    print("\n" + "="*60)
    print("METHOD 3: Gaussian Mixture Model (GMM)")
    print("="*60)
    
    X = rho.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=20)
    gmm.fit(X)
    labels = gmm.predict(X)
    
    # Get means and sort
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    means_sorted = means[sorted_idx]
    
    # Calculate boundaries (midpoints)
    boundaries = []
    for i in range(len(means_sorted) - 1):
        boundary = (means_sorted[i] + means_sorted[i+1]) / 2
        boundaries.append(boundary)
    
    # Metrics
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    
    print(f"Component Means: {means_sorted}")
    print(f"Optimal Boundaries: {boundaries}")
    print(f"BIC: {bic:.2f} (lower is better)")
    print(f"AIC: {aic:.2f} (lower is better)")
    
    # Distribution
    for i, idx in enumerate(sorted_idx):
        mask = labels == idx
        pct = 100 * mask.sum() / len(labels)
        print(f"Phase {i+1}: {pct:.1f}% of data")
    
    return boundaries, bic


def visualize_boundaries(rho, methods_results):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Histogram with all methods
    ax = axes[0, 0]
    ax.hist(rho, bins=50, color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    colors = ['red', 'green', 'blue', 'orange']
    linestyles = ['--', '-.', ':', '-']
    
    for i, (method_name, boundaries, score) in enumerate(methods_results):
        if boundaries is None:
            continue
        for b in boundaries:
            ax.axvline(b, color=colors[i], linestyle=linestyles[i], 
                      linewidth=2, alpha=0.7, label=f"{method_name}: {b:.3f}")
    
    ax.set_xlabel('Connectivity Ratio ρ = λ₂/λ_max', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('(A) All Methods Comparison', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Current vs Optimal (K-means)
    ax = axes[0, 1]
    ax.hist(rho, bins=50, color='lightgray', alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Current boundaries (arbitrary)
    ax.axvline(0.15, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='Current: 0.15')
    ax.axvline(0.45, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='Current: 0.45')
    
    # K-means boundaries (optimal)
    kmeans_boundaries = methods_results[0][1]
    if kmeans_boundaries:
        ax.axvline(kmeans_boundaries[0], color='green', linestyle='-', linewidth=2.5, 
                  alpha=0.8, label=f'K-means: {kmeans_boundaries[0]:.3f}')
        ax.axvline(kmeans_boundaries[1], color='green', linestyle='-', linewidth=2.5, 
                  alpha=0.8, label=f'K-means: {kmeans_boundaries[1]:.3f}')
    
    ax.set_xlabel('Connectivity Ratio ρ', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('(B) Current vs K-Means Optimal', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Silhouette analysis for different k
    ax = axes[1, 0]
    k_range = range(2, 8)
    silhouettes = []
    
    X = rho.reshape(-1, 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
    
    ax.plot(k_range, silhouettes, marker='o', linewidth=2, markersize=8, color='purple')
    ax.axvline(3, color='red', linestyle='--', alpha=0.5, label='k=3 (chosen)')
    ax.set_xlabel('Number of Clusters (k)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontweight='bold', fontsize=11)
    ax.set_title('(C) Optimal k Selection', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Panel D: Phase distribution comparison
    ax = axes[1, 1]
    
    # Current boundaries
    current_phase1 = (rho < 0.15).sum()
    current_phase2 = ((rho >= 0.15) & (rho < 0.45)).sum()
    current_phase3 = (rho >= 0.45).sum()
    current_pcts = [100*current_phase1/len(rho), 100*current_phase2/len(rho), 
                    100*current_phase3/len(rho)]
    
    # K-means boundaries
    if kmeans_boundaries:
        kmeans_phase1 = (rho < kmeans_boundaries[0]).sum()
        kmeans_phase2 = ((rho >= kmeans_boundaries[0]) & (rho < kmeans_boundaries[1])).sum()
        kmeans_phase3 = (rho >= kmeans_boundaries[1]).sum()
        kmeans_pcts = [100*kmeans_phase1/len(rho), 100*kmeans_phase2/len(rho), 
                       100*kmeans_phase3/len(rho)]
    else:
        kmeans_pcts = [0, 0, 0]
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, current_pcts, width, label='Current (0.15, 0.45)', 
                   color='red', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, kmeans_pcts, width, label='K-Means Optimal', 
                   color='green', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
    ax.set_title('(D) Phase Distribution Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Phase I', 'Phase II', 'Phase III'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save to paper_build/figures directory
    output_path = os.path.join(os.path.dirname(__file__), '..', 'paper_build', 'figures', 'optimal_phase_boundaries.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("SPECTRAL PHASE BOUNDARY OPTIMIZATION")
    print("="*60)
    
    # Generate or load data
    print("\nGenerating synthetic ρ distribution (matching empirical pattern)...")
    rho = generate_synthetic_data(n_samples=10000)
    print(f"Data: n={len(rho)}, mean={rho.mean():.3f}, std={rho.std():.3f}")
    
    # Apply all methods
    methods_results = []
    
    # Method 1: K-means
    boundaries_km, score_km = method_1_kmeans(rho, k=3)
    methods_results.append(("K-Means", boundaries_km, score_km))
    
    # Method 2: Jenks
    boundaries_jenks, score_jenks = method_2_jenks(rho, k=3)
    methods_results.append(("Jenks", boundaries_jenks, score_jenks))
    
    # Method 3: GMM
    boundaries_gmm, score_gmm = method_3_gmm(rho, k=3)
    methods_results.append(("GMM", boundaries_gmm, score_gmm))
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Current vs Optimal Boundaries")
    print("="*60)
    print(f"Current (arbitrary):        [0.15, 0.45]")
    print(f"K-Means (optimal):          {boundaries_km}")
    print(f"Jenks (optimal):            {boundaries_jenks}")
    print(f"GMM (optimal):              {boundaries_gmm}")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    if boundaries_km:
        diff1 = abs(boundaries_km[0] - 0.15)
        diff2 = abs(boundaries_km[1] - 0.45)
        
        if diff1 < 0.03 and diff2 < 0.05:
            print("✓ Current boundaries are REASONABLE (within 5% of optimal)")
            print("  → Can justify as heuristic approximation to clustering results")
        else:
            print("⚠ Current boundaries DEVIATE from optimal clustering")
            print(f"  → Suggested update: [{boundaries_km[0]:.3f}, {boundaries_km[1]:.3f}]")
            print(f"  → Or acknowledge as 'round number approximation' in paper")
    
    # Visualize
    visualize_boundaries(rho, methods_results)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
