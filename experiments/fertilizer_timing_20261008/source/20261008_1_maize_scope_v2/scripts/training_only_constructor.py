"""Use the already-qualified strict Parquet predicate constructor at every stage."""


def build_training(job, *, acceptance=False):
    from freeze_job import training_only_adapter
    adapter=training_only_adapter(job)
    # Diagnostic bypass is explicit and cannot issue a production gate.
    adapter.acceptance=bool(acceptance)
    return adapter
