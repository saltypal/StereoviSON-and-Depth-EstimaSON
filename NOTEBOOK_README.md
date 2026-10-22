# Stereo experiment presentation

Open Stereo_Vision_Experiment.html in a browser for the executed presentation. Open Stereo_Vision_Experiment.ipynb in VS Code and select .venv/Scripts/python.exe to Run All. Dependencies: requirements-notebook.txt. Keep IMG_0690.jpg and IMG_0692.jpg in the project root; numerical helpers live in src/refinement.py and src/pose.py.

Current pipeline: SIFT -> mutual matching -> local Lucas-Kanade patch refinement -> forward/backward check -> essential RANSAC -> pose recovery -> triangulation. Baseline: approximate full lens separation 0.201 m. Ground truth depth 0.28 m is used only for evaluation. Intrinsics remain approximate and lens distortion uncorrected.

Executed result: 127 mutual candidates, 77 pass local checks, 60 final triangulated points. New depth 29.809 cm (6.462% error), versus raw disparity 30.888 cm (10.314% error). Closer on this pair, but stability screening FAILS: subset depth spread 21.60%, P90 rotation variation 5.52 degrees. No validated metric distance is claimed. Filtering and pose changes prevent attributing the gain solely to subpixel adjustment.

Evidence: results/refined_20261021_025643_810193, including refinement_summary.json, refined_correspondences.csv, refined_triangulation.npz and figures. Each rerun creates a new results/refined_<timestamp> folder.

Earlier notebook versions and the previous standalone pose comparison are preserved under results/notebook_archive_20261021_025628, outside the active presentation.
