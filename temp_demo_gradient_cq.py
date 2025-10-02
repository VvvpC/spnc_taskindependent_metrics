"""Demo: run evaluate_heterogeneous_KRandGR for a gradient reservoir."""
import sys
import pathlib

# Ensure repository root is on sys.path so local modules import correctly when running directly
REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Morphology_Research.Reservoirs_morphology_creator import MorphologyConfig
from Morphology_Research.Reservoirs_morphology_evaluation import evaluate_heterogeneous_KRandGR
from formal_Parameter_Dynamics_Preformance import ReservoirParams


def build_demo_configuration():
    """Create reservoir parameters and a gradient morphology config for the demo."""
    reservoir_params = ReservoirParams(Nvirt=6, beta_prime=28, m0=0.008)
    # Keep the internal params dictionary in sync with the chosen Nvirt value
    reservoir_params.update_params(Nvirt=reservoir_params.Nvirt)

    config = MorphologyConfig(
        morph_type="gradient",
        beta_range=(24, 32),
        n_instances=3,
        random_seed=42
    )

    return reservoir_params, config


def main():
    reservoir_params, config = build_demo_configuration()
    # Empty list lets the evaluation helper create uniform weights for subreservoirs
    weights = []

    metrics = evaluate_heterogeneous_KRandGR(
        reservoir_params,
        config,
        weights,
        Nwash=6,
        seed=2025,
        threshold=0.05
    )

    print("Demo finished. Measured metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
