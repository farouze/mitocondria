# Submission notes and reusable text

These notes accompany the revised scientific master. No submission, repository deposit, journal contact or payment has been made. Confirm the final target before applying its complete template. The following original text summarizes completed analyses only.

## Optional expanded title for Journal of Microscopy

Representation Bias, Correction Transfer, and Resolution Sensitivity in Three-Dimensional Mitochondrial Morphometry

## Lay description

Mitochondria are tiny structures that help cells use energy. Scientists study their shapes to learn about cells, but a computer can give different measurements of the same mitochondrion depending on how its image is processed.

We compared several computer representations of mitochondrial shapes. In one public shape library, a representation based on points labeled inside or outside the object usually made mitochondria look slightly larger than the reference surface. The measurements still appeared to agree very well overall, showing why a single agreement score can hide a systematic difference.

A mathematical correction reduced the typical error on a previously unused batch from the same library. However, its estimated error limit did not work equally well for every shape. In a separate set of electron-microscope annotations, making the image grid coarser changed surface area and roundness more than volume. Changing the grid alignment also affected some results.

The practical lesson is that reliable-looking measurements need several checks: their average bias, their errors on individual objects, the shapes on which they fail, and their sensitivity to processing choices. This study helps researchers distinguish changes caused by a measurement procedure from changes that might reflect biology. It does not show whether differently shaped mitochondria produce the same amount of energy.

## Optional practitioner points

- Check object-level bias and errors alongside overall correlation or agreement.
- Validate a correction and its uncertainty bound separately, including difficult shape subgroups.
- Report grid spacing, alignment and voting rules when comparing surface area or sphericity.

## Optional short abstract for a 200-word limit

Mitochondrial measurements can change when the same object is represented or processed differently. We assessed representation bias and correction transfer using 2,720 development and 2,728 previously unused objects from a public mitochondrial shape library. A separate electron-microscopy analysis examined resolution sensitivity in 550 rat-cortex objects. Development occupancy volumes exceeded reference mesh volumes by 3.665% on average despite an intraclass correlation coefficient of 0.994. Boundary fits identified an outward label mismatch without isolating its cause. A frozen regression reduced second-batch median absolute percentage error from 3.481% to 0.664%. Coverage of its calibrated error bound declined from 96.2% internally to 92.1%, with 49.2% coverage in a pre-existing low-occupancy subgroup. In the separate resolution study, coarsening in-plane spacing from 8 to 24 nanometres changed median volume by −0.08%, surface area by −10.60%, and sphericity by +11.76%, despite strong rank preservation. Grid alignment and voting rules qualified volume stability. These findings distinguish numerical precision, absolute agreement, rank stability and uncertainty transfer. Correction transfer was tested within one resource; neither cross-pipeline portability nor biological invariance was established.

## Author-provided facts still needed for submission

Record the corresponding author and email; confirm full affiliations, ORCIDs and author order; specify actual funding and grant numbers, competing interests, contributions and acknowledgments. Confirm source-data terms and any preprint history. Record AI assistance accurately under the chosen journal's current disclosure policy. Do not copy generic “no funding” or “no conflicts” text without author confirmation.

## Release preparation

1. Execute the claimed workflow in a clean environment and attach its log; distinguish retained notebook results from newly reproduced analyses.
2. Include the input manifest, frozen/reconstructed artifact identities, processed tables, code, revised notebooks, instructions and result checks. The original raw archives are not redistributed in this package.
3. Choose an appropriate license for original code and text only after checking ownership and third-party dependencies; preserve third-party licenses.
4. Prepare a versioned repository or supplementary archive and cite its real URL once it exists. Keep any embargo or blind-review requirements consistent with the selected journal.
5. If moving Appendix A to a separate supplement, retain its exploratory status, all limitations and a main-text callout. The present master keeps it available for review.

The full clean-environment run and proposed stronger-validation experiments have not been performed in revision 9. No new causal or external-validation result should be added to the paper until those outputs exist.
