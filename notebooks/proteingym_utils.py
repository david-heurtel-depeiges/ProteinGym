# %%
import polars as pl

import numpy as np


def compute_bootstrap_standard_error_fast(
    df: pl.DataFrame, group_col="coarse_selection_type", number_assay_reshuffle=10000
):
    """
    Computes bootstrap standard error efficiently using Polars and Vectorized NumPy.

    Args:
        df: Polars DataFrame containing 'Selection Type' and model scores.
        number_assay_reshuffle: Number of bootstrap iterations.

    Returns:
        Polars DataFrame with model names and their standard errors.
    """
    # 1. Identify Model Columns (Numeric columns, excluding grouping key)
    # We use the selector to get only numeric columns
    model_cols = [c for c in df.columns if c != group_col and df[c].dtype.is_numeric()]
    n_models = len(model_cols)

    # 2. Partition data by Category (Selection Type)
    # This avoids repeated filtering/grouping
    partitions = df.partition_by(group_col, maintain_order=False)
    n_categories = len(partitions)

    if n_categories == 0:
        return pl.DataFrame()

    # 3. Accumulator for the Global Means
    # Shape: (n_iterations, n_models)
    # We sum the means from each category into this array
    global_sum_means = np.zeros((number_assay_reshuffle, n_models))

    # 4. Iterate over partitions (Categories)
    # Note: We loop over Categories (small N), not Bootstrap Iterations (huge N)
    for partition in partitions:
        # Convert partition to NumPy for high-performance matrix sampling
        # Shape: (n_samples_in_category, n_models)
        data_matrix = partition.select(model_cols).to_numpy()
        n_samples = data_matrix.shape[0]

        # A. Generate Random Indices for all 10,000 iterations at once
        # Shape: (number_assay_reshuffle, n_samples)
        # This simulates sampling with replacement
        bootstrap_indices = np.random.randint(
            0, n_samples, (number_assay_reshuffle, n_samples)
        )

        # B. Create the 3D Bootstrap Volume
        # Shape: (number_assay_reshuffle, n_samples, n_models)
        # This effectively selects the rows for every bootstrap iteration
        bootstrapped_data = data_matrix[bootstrap_indices]

        # C. Compute Mean across the 'samples' dimension (axis 1)
        # Shape: (number_assay_reshuffle, n_models)
        # This gives us the 10,000 means for THIS category
        category_means = bootstrapped_data.mean(axis=1)

        # D. Add to the global sum (Summing means across categories)
        # Since categories are independent, we can just sum their bootstrap vectors
        global_sum_means += category_means

    # 5. Final Calculation
    # Calculate the Macro-Average (Mean of Means) across categories
    global_averages = global_sum_means / n_categories

    # Calculate Standard Deviation across the 10,000 bootstrap samples
    # ddof=1 matches the pandas .std() default
    stds = global_averages.std(axis=0, ddof=1)

    # 6. Format Output
    return pl.DataFrame({"Model": model_cols, "Bootstrap_Standard_Error": stds})


def _collect_group_results(
    perf_by_assay: pl.DataFrame,
    category: str,
    value_col: str = "spearman",
):
    """Convert grouped aggregates into unified result rows.

    Args:
        df: Aggregated dataframe containing category and value columns.
        category_col: Column to use as category in output.
        type_label: Label to store in the "type" column.
        value_col: Column containing numeric values (default: "spearman").

    Returns:
        List of tuples shaped like (category, value, type, spread).
    """
    if perf_by_assay.is_empty():
        return []
    perf_by_category = perf_by_assay.group_by(category).mean()
    return (
        perf_by_category.select(pl.col(category).alias("category"), pl.col(value_col))
        .with_columns(
            pl.lit(category).alias("type"),
            pl.lit(None).alias("spread"),
        )
        .rows()
    )


def calculate_scores(df: pl.DataFrame, dms_file) -> pl.DataFrame:
    """Calculate weighted mean and bootstrap standard error for a set of spearman
    correlations according to official ProteinGym protocol. Assumes df has columns
    "spearman_correlations" and "assays".
    """

    with open(dms_file, "r") as f:
        metainfo: pl.DataFrame = pl.read_csv(f)

    assert isinstance(metainfo, pl.DataFrame)
    metainfo2 = metainfo.select(
        pl.col("DMS_filename").alias("assays"),
        pl.col("coarse_selection_type"),
        pl.col("UniProt_ID"),
        pl.col("MSA_Neff_L_category"),
        pl.col("taxon"),
    ).unique()

    uniprot_msa_lookup = metainfo.select(
        pl.col("UniProt_ID"), pl.col("MSA_Neff_L_category")
    ).unique()
    uniprot_taxon_lookup = metainfo.select(
        pl.col("UniProt_ID"), pl.col("taxon")
    ).unique()

    df = df.join(metainfo2, on="assays", how="left")
    perf_by_assay = df.group_by("UniProt_ID").agg(
        pl.col("spearman_correlations").mean().alias("spearman")
    )
    perf_by_assay = perf_by_assay.join(metainfo2, on="UniProt_ID", how="left")
    uniprot_function_level_average = perf_by_assay.group_by(
        "coarse_selection_type"
    ).mean()
    final_average = uniprot_function_level_average.mean()
    se = compute_bootstrap_standard_error_fast(perf_by_assay)
    results = []
    results.extend(_collect_group_results(perf_by_assay, "taxon"))
    results.extend(_collect_group_results(perf_by_assay, "MSA_Neff_L_category"))
    results.extend(_collect_group_results(perf_by_assay, "coarse_selection_type"))
    # perf_by_selection = perf_by_assay.group_by("coarse_selection_type").mean()
    # Put all results together, with column names "category", "value", "spread", "type"
    results.append(
        (
            "final",
            final_average["spearman"][0],
            "final",
            se["Bootstrap_Standard_Error"][0],
        )
    )
    return pl.DataFrame(
        results,
        schema=["category", "value", "type", "spread"],
        schema_overrides={"spread": pl.Float64},
    )


# %%
