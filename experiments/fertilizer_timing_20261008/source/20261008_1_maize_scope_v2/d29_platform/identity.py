"""Scientific identity and comparisons that do not hide changed conditions."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class RunIdentity:
    source_hash: str
    plant_activity_hash: str
    history_hash: str
    h1_hash: str
    land_model_id: str
    equations_hash: str
    mapping_id: str
    operator_hash: str
    sampling_support_hash: str
    parameters_hash: str
    objective_hash: str
    training_labels_hash: str
    experiment_role: str

    def as_dict(self):
        result = asdict(self)
        if any(not isinstance(v, str) or not v for v in result.values()):
            raise ValueError("every identity component must be explicit and non-empty")
        return result

    def digest(self):
        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class IdentityMismatch(ValueError):
    pass


def compare_identity(left: RunIdentity, right: RunIdentity, *, changed_factors=(), descriptive=False):
    a, b = left.as_dict(), right.as_dict()
    allowed = set(changed_factors)
    unknown = allowed - set(a)
    if unknown:
        raise ValueError(f"unknown identity fields: {sorted(unknown)}")
    changed = {name: {"left": a[name], "right": b[name]} for name in a if a[name] != b[name]}
    uncontrolled = sorted(set(changed) - allowed)
    if uncontrolled and not descriptive:
        raise IdentityMismatch("single-factor attribution rejected; uncontrolled changes: " + ", ".join(uncontrolled))
    return {"left_identity": left.digest(), "right_identity": right.digest(), "changes": changed,
            "uncontrolled_changes": uncontrolled,
            "comparison_type": "descriptive_only" if descriptive else "controlled_configuration_comparison",
            "unique_physical_causation_identified": False}


def four_corner_identity(identities):
    """Input and process swaps may vary, but history and observation support may not."""
    if len(identities) != 4:
        raise ValueError("four corners required")
    mutable = {"source_hash", "parameters_hash", "experiment_role"}
    reference = identities[0]
    return [compare_identity(reference, row, changed_factors=mutable) for row in identities[1:]]
