# Decision log

Team decisions about the data, the classes and the tools, with the reason for each. Config files such as `configs/categories.yaml` only hold *what* the code uses; the *why* lives here.

**Adding a decision:** add a new entry at the end of its section with the next free number. Never delete an entry; if it changes, set the old one to `Superseded by D-xx` and add a new one. Change this file through a PR like any other change.

Statuses: **Accepted**, **Superseded**, **Open** (still to be decided).

---

## Data sources

### D-01: Use TACO as the main dataset
- **Status:** Accepted
- **Date / by:** 2026-11-17, team
- **Decision:** Use TACO (Trash Annotations in Context) as the only main dataset. TrashNet and the Kaggle 12-class garbage set are no longer used.
- **Why:** TACO photos show real litter in real scenes, which is closer to our goal than single objects on clean backgrounds.
- **Consequence:** TACO is an object-detection dataset (boxes around objects, several objects per photo), so we crop each object into its own image (see D-12).
- **Source:** #7 discussion, PR #21

### D-02: Add the TU Wien organic waste dataset for food waste
- **Status:** Accepted
- **Date / by:** 2026-11-22, team
- **Decision:** Add the TU Wien "Image Dataset of Domestic Organic Waste and Non-Organic Contaminants" (CC BY 4.0, DOI 10.48436/27k90-dvw73). Only the 119 pure organic photos are used, as madaffald.
- **Why:** TACO has only 8 food-waste objects, and food waste is a major category in real life.
- **Alternatives considered:** Dropping madaffald like farligt_affald (D-07). Rejected because food waste is too important to leave out.
- **Known risk:** All TU Wien photos come from one household's bin, while the other classes come from TACO, so the model may learn the background instead of the food. To be checked in experiment #16. The 324 contaminated photos also contain food waste and could be cropped later using their masks.
- **Source:** PR #22

### D-03: Use both TACO annotation files, but validate and test on official data only
- **Status:** Accepted
- **Date / by:** 2026-11-22, team
- **Decision:** Use both `annotations.json` (official) and `annotations_unofficial.json`. Unofficial TACO goes into the training set only. Validation and test use official TACO plus the team-verified TU Wien photos.
- **Why:** More training data helps, but the unofficial labels are less checked, so keeping them out of val/test keeps our reported accuracy trustworthy.
- **Source:** PR #22

---

## Classes

### D-04: Food and drink cartons go into plast
- **Status:** Accepted
- **Date / by:** 2026-11-10, team (checked on affald.kk.dk)
- **Decision:** No separate carton class; cartons are mapped to plast.
- **Why:** Copenhagen collects cartons in the same bin as plastic, so a bin-level classifier does not need to tell them apart. The first AI-drafted class list had cartons as a separate class; checking kk.dk corrected this.
- **Source:** PR #19

### D-05: Predict bins, not litter types
- **Status:** Accepted
- **Date / by:** 2026-11-17, team
- **Decision:** TACO's 60 fine-grained labels (e.g. Clear plastic bottle, Plastic film, Crisp packet) are mapped to the Copenhagen bin they belong in. The mapping is `datasets.taco.map` in `configs/categories.yaml`.
- **Why:** The goal is to tell people which bin to use, not to name the exact item.
- **Source:** PR #21

### D-06: Keep restaffald as a real class
- **Status:** Accepted
- **Date / by:** 2026-11-22, team
- **Decision:** Keep residual waste as a class, even though it is a catch-all.
- **Why:** Without it, the model would have to put items like cigarette butts into a recycling bin, which is wrong and harmful.
- **Source:** PR #22

### D-07: Drop the farligt_affald (hazardous waste) class
- **Status:** Accepted
- **Date / by:** 2026-11-22, team
- **Decision:** No hazardous-waste class. TACO's Battery and Aerosol labels are dropped.
- **Why:** TACO has too few hazardous objects to split into train/val/test. The web-scraped datasets we found were unlicensed, full of duplicates and off-domain.
- **Known limitation:** The model cannot recognise batteries, aerosols, paint or bulbs and will sort them into another bin, which is wrong in Copenhagen. This must be reported in the evaluation (#11).
- **Supersedes:** the earlier plan to map Battery and Aerosol to farligt_affald.
- **Source:** PR #22

### D-08: Drop tekstilaffald (textiles)
- **Status:** Accepted
- **Date / by:** 2026-11-17, team
- **Decision:** No textile class. TACO's 7 Shoe objects are dropped.
- **Why:** TACO has no clothes or textiles, and a class with no images cannot be trained. Re-add if another dataset provides textile images.
- **Source:** PR #21

### D-09: Other Copenhagen schemes left out
- **Status:** Accepted
- **Date / by:** 2026-11-10, team
- **Decision:** Bleaffald (nappies), Elektronik, Stort indbo, Haveaffald and Træ are not modelled.
- **Why:** No image data, and most are outside the scope of photos of household items.
- **Source:** PR #19

### D-10: Reroute four previously dropped TACO labels
- **Status:** Accepted
- **Date / by:** 2026-11-22, team (verified on kk.dk by Mackey and Penelope)
- **Decision:** Paper cup, Pizza box and Wrapping paper → restaffald; Broken glass → glas.
- **Why:** These match the kk.dk sorting guidance and add useful data. The other dropped labels (e.g. Other carton, Meal carton, Rope & strings) stay dropped because they are ambiguous and have too few objects to be worth the debate.
- **Source:** PR #22

### D-11: Deposit (pant) bottles go to their material's bin
- **Status:** Accepted
- **Date / by:** 2026-11-22, team
- **Decision:** Bottles are mapped by material (plast, glas, metal), whether or not they carry a deposit.
- **Why:** A photo cannot show whether an item has a deposit, and the model only predicts waste categories.
- **Source:** PR #22

---

## Data preparation

### D-12: Crop each object and split by photo
- **Status:** Accepted
- **Date / by:** 2026-11-17, team
- **Decision:** Crop every annotated TACO object into its own image with one label. All crops from the same photo must go into the same train/val/test split.
- **Why:** Our model is a classifier (one image in, one class out). Crops from one photo share background and lighting, so splitting them across train and test would make test accuracy look better than it really is.
- **Alternatives considered:** Multi-label classification of whole photos. Rejected because it would mean rewriting #10, #11, #15 and #17.
- **Source:** #7 discussion, #8, #23

### D-13: Accept the class imbalance for now
- **Status:** Accepted
- **Date / by:** 2026-11-22, team
- **Decision:** Keep the data as it is (plast has about 2,000 objects; pap about 80) and deal with imbalance later.
- **Why:** Handling it is the subject of experiment #13.
- **Source:** PR #22

### D-14: Minimum crop size and padding
- **Status:** Open
- **Question:** Skip boxes smaller than 32 px (AI suggestion)? Add padding around each box (e.g. 10%)?
- **Facts so far:** At 32 px, restaffald in official TACO drops from about 970 to about 626 objects, mostly small cigarette butts.
- **To decide in:** #23

### D-15: Make crops square by padding, or stretch them?
- **Status:** Open
- **Question:** Pad each crop to a square before resizing to 224×224, or resize directly (which distorts long objects like straws and bottles)? Add vertical flip or rotation to training augmentation?
- **To decide in:** #9

---

## Tools

### D-16: PyTorch and Python 3.11
- **Status:** Accepted
- **Date / by:** 2026-11-10, team
- **Decision:** Use PyTorch with torchvision on Python 3.11.
- **Why:** Grad-CAM and embedding extraction need access to internal layers, which PyTorch hooks support directly, and torchvision provides the pretrained ResNet-18.
- **Alternatives considered:** Keras.
- **Source:** PR #18

### D-17: Choice of Grad-CAM implementation
- **Status:** Open
- **Question:** Write Grad-CAM by hand with PyTorch hooks, or use the `pytorch-grad-cam` library?
- **To decide in:** #14
