# Research justification for the VivaSense 50/35/15 answer-scoring weights

Audience: final-year project evaluators and academic examiners  
Date: 2 September 2026  
Scope: justification of `S_a = 0.50C + 0.35D + 0.15K`, where C is correctness, D is depth, and K is consistency.

## Direct answer

The exact vector 0.50/0.35/0.15 is not a universal constant prescribed by a published standard. It is defensible as a transparent, research-informed, expert-designed weighting scheme whose priority order is corroborated by educational-assessment research. The strongest claim is therefore not “research proves these exact numbers,” but “research supports the constructs and their ordering; the numerical weights operationalize that ordering and produce deliberately chosen score ceilings.”

## Claim-to-source ledger

1. Explicit analytic criteria and published weights improve transparency. Tomas et al. (2019) found that implicit holistic marking can diverge from intended learning outcomes and argued for explicit weighting. Their empirical criterion importances grouped approximately into 51-52% knowledge/understanding, 31-32% critical thinking, and 17% lower-order writing criteria. This is not the same construct set as VivaSense, but it independently corroborates a dominant direct-knowledge component, a substantial reasoning component, and a smaller supporting component. Source: https://doi.org/10.3389/feduc.2019.00089

2. Weighting should reflect both logical validity and empirical reliability. Wainer and Thissen (2004) state that component weights have reliability and validity implications and recommend considering both empirical reliability and logical validity evidence. Source: https://pubmed.ncbi.nlm.nih.gov/15446296/

3. Assessment weights should align with intended learning outcomes. Graves (2026) presents a quantitative framework connecting fair weighting to curriculum and learning-outcome alignment. Source: https://doi.org/10.14434/josotl.v26i1.37647

4. Correctness and depth represent different levels of evidence. Krathwohl's overview of revised Bloom's taxonomy distinguishes factual knowledge from increasingly complex processes including understanding, applying, analysing and evaluating. Source: https://www.psychology.mcmaster.ca/bennett/psy720/readings/m1/m1r1.pdf

5. Computing answers require multiple quality dimensions. Chen et al. (2020) found that code-explanation quality could not be represented adequately by one dimension; their validated rubric addressed correctness, abstraction and ambiguity, with median Krippendorff's alpha 0.775 and Cronbach's alpha 0.954. Source: https://doi.org/10.1145/3328778.3366879

6. Structured oral assessment is more defensible than unstructured judgement. Abuzied and Nabag's (2023) systematic review of 24 studies reported structured-viva reliability of alpha 0.75-0.80 in two settings versus alpha 0.50 for traditional viva. Source: https://doi.org/10.1186/s12909-023-04524-6

7. Consistency with submitted work is corroborating evidence of authenticity. Cao and Zahid (2025) describe viva voce as assessing submission quality while helping confirm authorship through questions tailored to the student's coursework. Source: https://kclpure.kcl.ac.uk/portal/en/publications/automated-viva-voce-using-generative-ai-for-student-coursework-au/

8. Evidence-Centered Design supports mapping claims to observable evidence rather than using an unexplained single score. Mislevy and Haertel (2006) frame assessment around the evidentiary argument it is intended to embody. Source: https://doi.org/10.1111/j.1745-3992.2006.00075.x

## Why 50% correctness

Correctness is the most direct evidence for the primary claim: the student understands the technical content of the project. Assigning half of the total score makes factual validity dominant without making the rubric one-dimensional. It is also close to the 51-52% combined importance of knowledge-and-understanding criteria observed by Tomas et al. This is corroboration, not a claim that their categories are identical to ours.

The weight also creates a deliberate ceiling. If correctness is zero, perfect depth and consistency can produce at most 0.50. A fluent, detailed and internally consistent but false answer therefore cannot obtain a high mark.

## Why 35% depth

Depth captures explanation, mechanisms, justification, trade-offs and application. Bloom's taxonomy supports separating factual knowledge from higher-order cognitive processes. Thirty-five percent is large enough to prevent a shallow correct answer from being treated as excellent while remaining below correctness because elaborate reasoning cannot repair false technical claims.

If depth is zero, perfect correctness and consistency produce at most 0.65. The model therefore distinguishes recall from defensible project understanding.

## Why 15% consistency

Consistency checks whether the answer coheres with the student's report, code and earlier answers. This supports authentication and detects contradictions, but it is indirect evidence. It can also be affected by speech-to-text errors, ambiguous questions, changed assumptions, or limited prior transcript. Its weight is therefore intentionally capped.

If consistency is zero, perfect correctness and depth can still produce 0.85. A technically strong answer is penalized, but a noisy corroborating signal cannot overwhelm direct evidence.

## Derivation logic

The design can be described in two stages:

1. Allocate 50% to the primary direct evidence: correctness.
2. Divide the remaining 50% in a 70:30 ratio between direct evidence of understanding (depth) and corroborating evidence (consistency), yielding 35% and 15%.

This produces the intended ordering C > D > K and simple, inspectable score consequences. It is a constrained expert judgement, consistent with literature recognizing expert-derived percentage weights as an accepted initial method, followed by empirical validation.

## What must not be claimed

Do not say that a paper discovered or mandated the exact 50/35/15 vector. No located credible source does so for this exact technical-viva construct. The appropriate description is “literature-informed initial weights” or “expert-informed weights corroborated by assessment research.”

## Recommended validation

Have at least two qualified examiners independently score a frozen set of answers. Compare 50/35/15 with equal weighting and fitted non-negative weights using held-out validation. Report inter-rater agreement, correlation with examiner holistic marks, mean absolute error, and sensitivity to +/-5 percentage-point changes. If correctness is intended to be non-compensable, also test a minimum-correctness gate because a weighted sum is otherwise a compensatory model.

## Presentation-ready defence

“The weights were not taken as an unexplained universal constant. They were defined as an explicit analytic combination rule aligned with the purpose of a technical viva. Correctness receives 50% because it is the most direct and necessary evidence of technical competence. Depth receives 35% because Bloom's taxonomy and oral-assessment research distinguish recall from explanation, application and critical reasoning. Consistency receives 15% because coherence with the report, code and previous answers strengthens authenticity, but it is indirect and more vulnerable to transcription and context errors. The resulting ceilings are intentional: an answer with zero correctness cannot exceed 50%, an answer with zero depth cannot exceed 65%, while an inconsistency alone cannot reduce an otherwise correct and deep answer below 85%. Educational-assessment research supports explicit expert-derived percentage weights and subsequent empirical checking; notably, one authentic marking study found roughly 51-52% importance for knowledge and understanding, 31-32% for critical thinking and 17% for secondary criteria. We therefore describe 50/35/15 as a transparent, literature-informed initial weighting, to be calibrated against independent examiner judgements rather than as a universal ratio.”

## References

Abuzied, A. I. H., & Nabag, W. O. M. (2023). Structured viva validity, reliability, and acceptability as an assessment tool in health professions education: A systematic review and meta-analysis. BMC Medical Education, 23, 531. https://doi.org/10.1186/s12909-023-04524-6

Cao, H., & Zahid, S. (2025). Automated Viva Voce Using Generative AI for Student Coursework Authentication. In 2025 International Conference on Educational Technology and Artificial Intelligence. ACM. https://kclpure.kcl.ac.uk/portal/en/publications/automated-viva-voce-using-generative-ai-for-student-coursework-au/

Chen, B., Azad, S., Haldar, R., West, M., & Zilles, C. (2020). A validated scoring rubric for explain-in-plain-English questions. Proceedings of the 51st ACM Technical Symposium on Computer Science Education, 563-569. https://doi.org/10.1145/3328778.3366879

Graves, J. (2026). Optimal Assessment Weighting: How much should I weight that final exam? Journal of the Scholarship of Teaching and Learning, 26(1). https://doi.org/10.14434/josotl.v26i1.37647

Krathwohl, D. R. (2002). A revision of Bloom's taxonomy: An overview. Theory Into Practice, 41(4), 212-218. https://doi.org/10.1207/s15430421tip4104_2

Mislevy, R. J., & Haertel, G. D. (2006). Implications of Evidence-Centered Design for Educational Testing. Educational Measurement: Issues and Practice, 25(4), 6-20. https://doi.org/10.1111/j.1745-3992.2006.00075.x

Tomas, C., Whitt, E., Lavelle-Hill, R., & Severn, K. (2019). Modeling holistic marks with analytic rubrics. Frontiers in Education, 4, 89. https://doi.org/10.3389/feduc.2019.00089

Wainer, H., & Thissen, D. (2004). Recommendations for assigning weights to component tests to derive an overall grade. Medical Education, 38(8), 861-867. https://pubmed.ncbi.nlm.nih.gov/15446296/
