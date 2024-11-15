# ArcticTraining

## TODOs:
- High priority
  - Add CI (focus on user-facing functionality)
  - Add logging / fix multiprocessing output
  - Add documentation (how to run current projects / trainers, how to add your own)
  - CLI + yaml inputs (CLI + Python frontends)
- Low priority
  - Add WandB support
  - Issue templates
  - Fully implement more default callbacks (e.g., eval?)
  - Refactor data loader to better accomodate collator
  - Refactor TrainerState... do we want to keep this? What attributes should be set in the trainer vs trainerstate objeect?
  - Extend type hints (remove `Any` where possible)
