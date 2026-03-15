from .utils import (
    to_chat,
    compute_mean_activations,
    diffmean,
    generate_with_steered_model,
    select_candidate_layers,
)

try:
    from .orthogonalization import (
        compute_reference_activations,
        orthogonalize_direction,
        orthogonalize_direction_from_data,
    )
except ImportError:
    # Some checkouts do not include the optional orthogonalization helpers.
    pass


