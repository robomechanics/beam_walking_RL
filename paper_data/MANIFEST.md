# Paper data

`specialist_narrow_20260914/` is the current data set: narrow-stance trot and
walk specialists (stance width 0.05-0.30 m), trained without disturbances, with
every evaluation exported to CSV and the five paper figures. Its `README.md`
describes the measurements and the claims each figure supports.

The earlier wide-stance data set (`specialist_20260914/`: stance width
0.10-0.50 m specialists and the trot push fine-tune) is no longer in the tree.
It is preserved at commit `6c3dfd0`:

    git checkout 6c3dfd0 -- paper_data/specialist_20260914

Earlier curated data (seed-2 paper policy, seed-3 adaptive selector, seed-4
high-duty walk, narrow-stance lineage and its push study) and the generated
figures are preserved at the git tag `pre-specialist-data`:

    git checkout pre-specialist-data -- paper_data PAPER_GRAPHS
