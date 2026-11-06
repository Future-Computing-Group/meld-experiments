# Claim-pair annotation rubric

You label pairs of short statements. For each pair, decide one verdict by the ordered steps below, then a subtype where the verdict has subtypes, then the overlap flag. Judge only the two texts as written: do not use outside knowledge about which statement is true, and do not assume any context beyond the two texts. The two texts have equal standing and their order carries no meaning. In the rationale, refer to the statements by their content, never as "A", "B", "first" or "second".

## Verdicts

Apply the steps in order and stop at the first one that holds.

Step 1, MERGE: the two statements assert the same proposition about the same entities, scope, time and quantity, so that keeping only one of them loses nothing. Differences in wording, voice, spelling, typos and plain synonyms (cutting and slicing) do not matter. A change of who did what to whom, of a number, date, name or place, an inserted or removed negation, or an added or removed detail that changes what is asserted rules out MERGE.

Step 2, CONFLICT: read as referring to the same time and place, the two statements cannot both be true. Subtypes:
- SUPERSESSION: the same attribute of the same entity has a different value because it changes over time (a count, a status, a current holder, a total to date); each statement can have been true at its own time.
- CORRECTION: a fixed fact differs (a birth date, an origin, who did what, a relation, a negation); one of the two statements is wrong.
- SCOPE: the statements clash as written, but the clash could come from an unstated difference of scope, population or condition, and the texts do not say which scope each one means.

Step 3, RELATE: both statements can be true together and they are about the same entity, event or topic. Subtypes:
- ELABORATION: one statement adds a detail (an attribute, a modifier, an event, a figure) to what the other says, or the two add compatible details about the same thing.
- SCOPE: the statements make the same kind of claim over a different range (a region, a population, a period, a set), and both can hold.

Step 4, NO_RELATION: none of the above; the statements are about different entities, events or topics.

## Overlap flag

Set overlap_flag to true when the two statements share most of their words but differ in meaning: swapped roles or places, a changed number, date, name or entity, or an inserted negation. It is never true with MERGE.

## Confidence

An integer from 1 (a guess) to 5 (certain under this rubric).

## Output

Answer with one JSON object and nothing else, of this form:

{"labels": [{"item_id": "<the item's id>", "verdict": "<verdict>", "subtype": "<subtype or null>", "overlap_flag": false, "confidence": 4, "rationale": "<one sentence>"}]}

Give exactly one entry per item and every item once. The verdict is one of the verdicts listed for the batch. The subtype is null for MERGE and NO_RELATION and one of that verdict's subtypes for CONFLICT and RELATE. The rationale is one short sentence.

## Worked examples

<!-- dev:324db608b069f179 -->
- "A person is slicing some mushrooms" and "A person is cutting some mushrooms": MERGE, subtype null, overlap_flag false. Slicing and cutting are synonyms here; nothing is lost by keeping one.

<!-- dev:fd83d6a7c89d3f90 -->
- "Parched is the story of four women residents of a dessert village of Rajasthan, India." and "Parched is the story of four women residents of a desert village of Rajasthan, India.": MERGE, subtype null, overlap_flag false. The only difference is a spelling fix.

<!-- dev:54b18991ed1f911b -->
- "Two people are sitting on a white bench" and "Two people are sitting on a bench": RELATE, ELABORATION, overlap_flag false. Both can be true; one adds the colour of the bench, so keeping only the shorter one would lose a detail.

<!-- dev:1f10fedf9499a6f6 -->
- "He is regarded as one of the best box-to-box midfielders due to his excellent dribbling skills and ability to maintain possession." and "He is regarded as one of the best box-to-box midfielders in Europe due to his excellent dribbling skills and ability to maintain possession.": RELATE, SCOPE, overlap_flag false. The same claim is made over a different range (in Europe versus without limit), and both can hold.

<!-- dev:4f803c53b84e3d42 -->
- "Shaffer Chimere Smith (born October 18, 1979), better known by his stage name Ne-Yo, is an American singer, songwriter, record producer, actor and dancer." and "Shaffer Chimere Smith (born October 18, 1982), better known by his stage name Ne-Yo, is an American singer, songwriter, record producer, actor and dancer.": CONFLICT, CORRECTION, overlap_flag true. A birth year is a fixed fact, so one of the two years is wrong.

<!-- dev:2a5a6c36bef24b2b -->
- "born May 12, 1979) is an former American football wide receiver of the National Football League (NFL)." and "born May 12, 1979) is an American football wide receiver for the Baltimore Ravens of the National Football League (NFL).": CONFLICT, SUPERSESSION, overlap_flag true. Whether the player is active is a status that changes over time; the statement that he is a former receiver replaces the one that he plays for a team.

<!-- dev:cf0b5b3b517f0f66 -->
- "The Slatina River is a tributary of the Cochirleanca River in Romania" and "The river Cochirleanca is a tributary of the Slatina river in Romania.": CONFLICT, CORRECTION, overlap_flag true. The two rivers swap roles, and two rivers cannot each be a tributary of the other.

<!-- dev:7d547311f3e60b88 -->
- "A woman is chopping an onion" and "A woman is not chopping an onion": CONFLICT, CORRECTION, overlap_flag true. The inserted negation makes the statements incompatible.

<!-- dev:a865a1925ccd505c:a dev:14d56ecde2dded37:b -->
- "A woman is chopping a red onion" and "Two people are looking at a gold statue": NO_RELATION, subtype null, overlap_flag false. Different people, actions and objects.

## Adjudication

This section applies only when the items carry votes. Each item lists the anonymous votes of independent annotators who disagreed on it or were inconsistent across the two orders of the texts: a verdict, a subtype, an overlap flag, a one-sentence rationale, and whether that annotator saw the texts in the order shown or reversed. The votes are evidence, not a ballot: apply the rubric to the texts yourself, and a majority can be wrong. When the texts support two verdicts and the rubric does not decide between them, answer AMBIGUOUS with subtype null and overlap_flag false. AMBIGUOUS is allowed only on items that carry votes.
