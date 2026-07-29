## Take-aways

- **The answer is regime-dependent, and that is the finding.** On cluttered,
  boxed scenes (Visual Genome) MaxPatch is the best strategy; on easy, centred,
  boxless images (Caltech-101) it is the worst. Pooling the two into one number
  hides the effect — the strategy choice has to be made against the *content*,
  not in the abstract.
- **On cluttered scenes, *where* the object is beats *what* global vector you
  have.** The largest gap in the study is region-scoring vs whole-image scoring,
  not MaxPatch vs MaxHAC. A DINOv3 whole-image (CLS) detector is the *worst* arm
  on Visual Genome — below SigLIP — while the same embedder with region votes is
  the *best*. When the target is a small part of a busy image, a representation
  that can point at the object wins decisively.
- **MaxPatch's win is a recall win, and it is scale-driven.** Against MaxHAC it
  lowers the miss rate (FNR) more than the false-alarm rate, and the advantage
  concentrates on sub-leaf-scale objects (Figure 4): below the ~8 %-area leaf
  scale the tree's smallest pooled candidate already blends the object with its
  surroundings, while a raw patch stays a near-pure object sample. For search,
  missing real matches is the expensive failure, so the recall win is the one
  that matters.
- **MaxPatch's failure is calibration, not ranking.** On easy data it still
  ranks perfectly (AP = 1.0) but its 196-way max-pool compresses scores so the
  trained threshold under-recalls (FNR 0.69 at FPR 0). This is important for
  productisation: MaxPatch does not need a better *representation* on easy
  content, it needs a max-pool-aware *threshold*. MaxHAC's smoothed pool avoids
  the problem entirely, which is why it is the safer generalist.
- **Effort changes the answer.** MaxPatch and MaxHAC are statistically tied for
  the first ~50 votes and only diverge as the session goes on — so the strategy
  choice matters most for the Autopilot power-user who keeps refining, and
  barely at all for a few-vote drive-by.
- **The tree is deletable exactly where MaxPatch wins.** MaxHAC beats
  whole-image on clutter, so its pooled regions do carry signal — but they carry
  *less* than the raw patches they are pooled from on the small-and-medium
  objects that dominate cluttered images. Where region votes are the workflow,
  the HAC build is a cost that is not paid back.
