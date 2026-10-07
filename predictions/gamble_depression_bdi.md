# Predictions: Depression and the two-timescale model (Rutledge gamble data, BDI-II subset)

- Original private commit: `fa830ae33a0a7dcdfebd47952079ef966e9465d5`
- Author timestamp: 2026-10-01T08:57:05-04:00
- Committer timestamp: 2026-10-01T08:57:05-04:00
- Source file in that commit: `reviews/GAMBLE_BDI_TEST.md`

The text below is reproduced verbatim from the file as it stood in that commit, before the analysis was run. The commit sits in the project's private version-control history, which also holds confidential peer-review material, so the commit itself cannot be published. See README.md in this folder.

---

# Depression and the two-timescale affect model on the Rutledge gamble data

## Predictions, written before any BDI analysis (2026-10-01 08:57:05 -0400)

Data: the depData set in data_raw/rutledge_gbe/Rutledge_GBE_risk_data_TOD.mat (1,858 participants
of The Great Brain Experiment app who also completed the Beck Depression Inventory, BDI total 0-63).
Model: the joint happiness and choice model of analysis/joint_gamble (two-timescale readout with
three channels, prospect-theory choice coupled to mood). Per-person parameters are estimated with
group parameters fixed from the 46,204-participant fit.

Higher BDI goes with:
- (a) a lower slow mood baseline (person baseline a);
- (b) a slower mood timescale (higher mood persistence rho);
- (c) a smaller present-channel weight (reduced reward sensitivity, anhedonia), or a larger
  negative response to losses than to wins;
- (d) a weaker or more negative forward-channel weight;
- (e) weaker coupling of mood to choice optimism (smaller eta).

Prior result to compare against: Rutledge, Moutoussis, Smittenaar, Zeidman, Taylor, Hrynkiewicz,
Lam, Skandali, Siegel, Ousdal, Prabhu, Dayan, Fonagy and Dolan (2017), "Association of neural and
emotional impacts of reward prediction errors with major depression", JAMA Psychiatry 74(8),
790-797, doi:10.1001/jamapsychiatry.2017.1713. In the smartphone sample (n = 1,833) depression
symptom severity correlated with the baseline mood parameter of the happiness equation and was
not associated with a reduced emotional impact of reward prediction errors. So (a) is expected to
replicate, and (c) is expected to fail unless the two-timescale decomposition separates something
the happiness equation did not.
