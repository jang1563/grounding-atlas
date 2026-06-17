# 3D-structure rung: hERG from raw XYZ coordinates (the encoding-limited candidate)

*Results. 2026-06-12. `eval/frontier_output_panel.py` (structure3d) + `eval/activation_arm_3d.py` (8B, pending). Same hERG molecules as the SMILES rung, presented as raw 3D atomic coordinates. The 8th rung, and the candidate for the genuine encoding-limited anchor the modality ladder lacked. No em dashes.*

## The rung and the within-entity contrast

The modality ladder found no clean encoding-limited point: coarse hERG is surface-decodable from every STRING representation (SMILES char-n-gram, DNA k-mer, MS m/z, image OCSR). 3D structure is the candidate because the property is in the 3D geometry, and recovering it from raw coordinates requires geometry parsing a forward pass cannot do. The molecule is rendered as XYZ atomic coordinates (RDKit ETKDG conformer, n=400 balanced hERG), e.g. "C 1.23 -0.57 2.89 / N -1.35 0.68 -1.23 / ...", and the SAME molecule's SMILES is the web-rich string counterpart, a within-entity contrast like variant text-vs-seq.

## Ceiling and frontier output

| | AUROC |
|---|---|
| ceiling (Morgan FP on the SMILES, the structure the coords encode) | 0.825 |
| ceiling (3D-shape descriptors only: asphericity, NPR, RoG) | 0.612 |
| output opus-4.8 (XYZ -> hERG) | 0.539 |
| output sonnet-4.6 | 0.515 |
| output haiku-4.5 | 0.487 |

The frontier OUTPUT is at chance across all scales (0.487 to 0.539), no scale trend. So no frontier model can read hERG from raw 3D coordinates. The same hERG molecules as a SMILES string are read at 0.45 to 0.63 (8B to sonnet), so within one entity the web-rich string is partly groundable while the web-zero coordinates are not: 3D coordinates join MS m/z and the anonymized single-cell IDs as a web-zero, scale-invariant-at-chance representation.

## The encoding-limited question (8B activation, pending)

Output-at-chance is necessary but not sufficient for encoding-limited; the decider is the 8B ACTIVATION on the XYZ text against the Morgan ceiling (0.825):
- activation near chance (< ~0.62) = ENCODING-LIMITED: the model cannot even form the structure from coordinates, the first such anchor (every prior rung encoded the coarse signal from surface tokens at 0.73 to 0.88).
- activation moderate (encodes the coordinate surface pattern) = expression-limited like the others, and coarse hERG is surface-decodable even from coordinate tokens.
RESULT (8B): ceiling (Morgan) 0.826, ACTIVATION 0.669, OUTPUT 0.490. So the 8B encodes hERG from the XYZ coordinates at 0.669, ABOVE chance but FAR below the Morgan ceiling, an encoding gap of 0.156, the LARGEST among the hERG representations (SMILES 0.038, image 0.096, MS 0.096, 3D-coords 0.156). Reading: the 8B picks up the ATOM-COMPOSITION surface signal (the element symbols C, N, O, Cl in the coordinate list give the molecular formula, which correlates with hERG via size and lipophilicity), but it cannot extract the 3D GEOMETRY (which atoms are bonded, the shape) that the forward pass would have to parse. So 3D coordinates are the most encoding-limited of the hERG representations, but NOT cleanly encoding-limited: even raw coordinate tokens carry the coarse hERG signal to the hidden states via atom counting.

The encoding-gap trend (SMILES 0.038 -> image 0.096 -> MS 0.096 -> 3D-coords 0.156) confirms that the harder the representation is to parse, the larger the encoding gap, but a CLEAN encoding-limited anchor (activation near chance) still did not materialize, because coarse hERG is partly recoverable from atom composition in any representation. A genuine encoding-limited point needs a FINE property where surface atom-counting fails and the 3D geometry is essential (specific-pocket binding affinity, the Boltz-2 version of this rung), confirming yet again that encoding-limited is property-granularity-dependent, not representation-dependent.

## Caveats

n=400, single conformer per molecule (ETKDG, not the bioactive pose), hydrogens included, coordinates rounded to 0.01 A. The ceiling is the 2D Morgan probe (0.825) since the coordinates encode the connectivity; a true 3D-GNN affinity specialist (Boltz-2 class) would be the stronger ceiling for a binding-affinity property (the deep-research-recommended version of this rung). This pilot uses hERG for direct comparability to the SMILES/image/MS rungs.

## Reproduce

Data: `signal/structure3d/herg_xyz.csv` (RDKit ETKDG on the hERG matched set). Output: `PANEL_RUNGS=structure3d python eval/frontier_output_panel.py`. Activation: `sbatch run_activation_3d_cayuga.sh`.
