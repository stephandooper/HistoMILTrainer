Converted data splits

Created with: python scripts/convert_data_splits.py INPUT [--output OUTPUT]
The CV dataset and boolean/descriptor split files are under cv/. No case_id is needed.
CV test membership intentionally matches validation membership for each fold, allowing
the trainer's final extra pass. This is not an independent evaluation. Do not run
histomil-splits: it would regenerate folds. All eight external test cohorts are retained
separately under test_sets/, including OOD metadata. For independent evaluation, run
prediction separately for each saved fold checkpoint and test CSV, then join predictions
to labels by slide_id to calculate metrics; the current predictor does not calculate them.
Feature files must be named
<slide_id without extension>.h5 and supplied through the appropriate feature directory.
In the supplied DLBCL archive, joint_2_test combines internal_test, hovon_test and
s05_test, while joint_1_test combines joint_2_test and stanford_test. These cohorts
overlap and are retained separately. Stanford contains only one positive slide.
