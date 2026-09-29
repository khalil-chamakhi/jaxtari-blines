# Contributions

Group 5, Topic 25 — Agent57 in JAXtari

The project was assigned to three members. One member did not participate, so the
work was redistributed between the two remaining members during the project.

## Khalil Chamakhi
- Stages 1, 2 and 4: recurrent replay learner (R2D2), the NGU intrinsic-reward
  stack (embedding, episodic memory, RND, arms), and the Agent57 meta-controller.
- Evaluation: the separate recurrent evaluator, and the fix for evaluating an
  arm-conditioned policy under training-like inputs.
- Infrastructure: configuration files, run scheduling across the shared GPUs,
  result collection, report and pull request.
- Integration of the split-Q stage with the rest of the agent.

## Aman Allah Guerfel
- Stage 3 (split-Q): design, implementation, testing and verification of separate
  extrinsic and intrinsic Q-networks and the split learner wiring, which forms the
  basis of the stage used in the final results.
