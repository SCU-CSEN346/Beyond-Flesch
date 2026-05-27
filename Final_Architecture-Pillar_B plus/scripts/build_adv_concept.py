"""
Build the AdvConcept-50 evaluation set.

The set is hand-written to separate what a text looks like from what it is
actually teaching. A short question can require a hard concept, and a long
sentence can still describe an elementary idea. That is the failure mode this
benchmark is meant to catch.

The labels are curriculum labels, not readability scores. Each row keeps a short
reason so the choice is auditable later.
"""

import csv
import os
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR     = os.path.join(_PROJECT_DIR, 'data')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')


def _display_path(path):
    return os.path.relpath(path, _PROJECT_DIR)


# Each row is:
#   (text, true_level, surface_complexity, category, reasoning)
#
# The examples below cover short/hard concepts, long/easy concepts, paired
# controls, and sanity-check rows where surface and concept agree.

SEED_ROWS = [
    # Short wording, harder concept.
    ("What is mitosis?",
     "high", "easy", "surface_easy_concept_hard",
     "Mitosis is taught in US high-school biology (grades 9-10, NGSS HS-LS1-4). 3-word question is grammatically trivial; concept requires understanding of cell-cycle phases."),
    ("Define angular momentum.",
     "high", "easy", "surface_easy_concept_hard",
     "Angular momentum is a physics-class topic (grades 11-12). Two short words but conceptually requires vectors and rotational kinematics."),
    ("What is fertilization?",
     "middle", "easy", "surface_easy_concept_hard",
     "Plant/animal fertilization is taught in middle-school life science (grades 6-8, NGSS MS-LS3-1, MS-LS3-2). 3-word question, simple vocab."),
    ("Calculate the derivative of x squared.",
     "high", "easy", "surface_easy_concept_hard",
     "Differential calculus is grade 11-12 (AP Calculus). Question phrased in elementary words but requires calculus."),
    ("What is photosynthesis?",
     "middle", "easy", "surface_easy_concept_hard",
     "Photosynthesis is core middle-school life science (grades 6-8, NGSS MS-LS1-6). Single short word question."),
    ("Solve the quadratic equation x squared minus 5x plus 6 equals zero.",
     "high", "medium", "surface_easy_concept_hard",
     "Quadratics by factoring is grade 9-10 algebra (Common Core HSA-REI.B.4). Words are simple, math is not."),
    ("What are ions?",
     "high", "easy", "surface_easy_concept_hard",
     "Chemical bonding and ions taught in high-school chemistry (grade 10-11). 3-word question."),
    ("What is the speed of light?",
     "high", "easy", "surface_easy_concept_hard",
     "Conceptually elementary as a fact, but its role in special relativity / EM waves is high-school physics. We classify by concept depth not factual recall."),
    ("Explain the Krebs cycle.",
     "high", "easy", "surface_easy_concept_hard",
     "Cellular respiration / Krebs cycle is AP Biology / high school (NGSS HS-LS1-7). Phrasing is a 3-word imperative."),
    ("What is a derivative?",
     "high", "easy", "surface_easy_concept_hard",
     "Calculus, grade 11-12. Single short question."),
    ("Define entropy.",
     "high", "easy", "surface_easy_concept_hard",
     "Thermodynamics / entropy is high-school physics or chemistry. Two-word imperative."),
    ("What is DNA?",
     "middle", "easy", "surface_easy_concept_hard",
     "DNA introduced in middle-school life sciences (grades 6-8, NGSS MS-LS3-1). 3-word question."),
    ("Find the integral of x.",
     "high", "easy", "surface_easy_concept_hard",
     "Calculus integration is grade 11-12. Words simple."),
    ("What are tectonic plates?",
     "middle", "easy", "surface_easy_concept_hard",
     "Plate tectonics is middle-school earth science (grades 6-8, NGSS MS-ESS2-3). Short question, simple vocabulary."),
    ("What is a chemical bond?",
     "high", "easy", "surface_easy_concept_hard",
     "Bonding / molecular structure is high-school chemistry (grade 10-11). Brief phrasing."),

    # Short math prompts that need later-grade techniques.
    ("Use Cramer's rule.",
     "high", "easy", "surface_easy_concept_hard",
     "Cramer's rule (linear systems via determinants) is grade 11-12 precalculus / linear algebra preview. 3 words."),
    ("Apply the chain rule.",
     "high", "easy", "surface_easy_concept_hard",
     "Chain rule of differentiation is grade 11-12 AP Calculus. Short imperative."),
    ("Factor x squared minus 9.",
     "high", "easy", "surface_easy_concept_hard",
     "Difference of squares — Algebra I, grade 8-9 (Common Core 8.EE.A.2 / HSA-SSE)."),
    ("Compute sine of pi over 6.",
     "high", "easy", "surface_easy_concept_hard",
     "Trig values are high-school precalculus (grade 11)."),
    ("Find the limit as x goes to zero of sin x over x.",
     "high", "medium", "surface_easy_concept_hard",
     "Classic limit, AP Calculus (grade 11-12)."),

    # Long wording, elementary concept.
    ("The small brown dog ran across the bright green grass in the sunny park while his happy owner cheerfully waved at him from the wooden bench under the tall leafy tree, and then the dog stopped suddenly because he saw a beautiful red ball lying on the soft grass nearby.",
     "elementary", "hard", "surface_hard_concept_easy",
     "Long, multi-clause sentence with several adjectives, but the *concept* — a dog seeing a ball in a park — is elementary-level (K-2 typical content)."),
    ("Yesterday afternoon during the bright sunny weather, the curious little girl named Sarah carefully picked up the colorful round apple from the green wicker basket on the wooden kitchen table and then thoughtfully placed it inside her favorite blue lunchbox before walking quietly out the front door of her cozy white house.",
     "elementary", "hard", "surface_hard_concept_easy",
     "Lots of subordinate clauses and descriptive adjectives, but the content — a girl putting an apple in a lunchbox — is K-2."),
    ("The fluffy orange cat with the long whiskers and the bright green eyes sat comfortably on the soft warm rug in the middle of the bright sunny living room next to the wide open window and watched the colorful birds flying happily through the clear blue sky.",
     "elementary", "hard", "surface_hard_concept_easy",
     "K-2 content (a cat watching birds) dressed in a long sentence."),
    ("During the wonderful springtime morning when the cheerful birds were singing their sweet melodious songs and the gentle warm breeze was softly blowing through the tall green trees, the young curious children eagerly ran outside to happily play their favorite games in the beautiful flower-filled meadow.",
     "elementary", "medium", "surface_hard_concept_easy",
     "Standard children-at-play content, but the prose is florid. Concept is K-3."),
    ("Inside the cozy old wooden barn that stood proudly at the back of the large green farm, the friendly brown horses with their long flowing manes happily munched their crunchy yellow hay while gently swishing their long tails to chase away the buzzing summer flies.",
     "elementary", "hard", "surface_hard_concept_easy",
     "K-3 farm scene; long descriptive sentence."),
    ("The bright yellow school bus with the loud honking horn slowly drove down the long winding road past the tall leafy trees and the colorful flower gardens until it finally stopped at the corner where the cheerful smiling children were patiently waiting with their heavy backpacks.",
     "elementary", "medium", "surface_hard_concept_easy",
     "Riding the school bus — K-3."),

    # Paired controls: same broad topic, different curriculum level.
    ("Plants need water and sunlight to grow.",
     "elementary", "easy", "surface_matches_concept",
     "K-2 plant biology fact. Both content and surface are elementary."),
    ("During photosynthesis, plants convert carbon dioxide and water into glucose using sunlight, with chlorophyll absorbing light primarily in the red and blue wavelengths.",
     "middle", "medium", "surface_matches_concept",
     "Middle-school biology (NGSS MS-LS1-6). Surface complexity matches."),
    ("Photosynthesis involves the light-dependent reactions in the thylakoid membrane, where photosystems II and I drive electron transport to generate ATP and NADPH, which are subsequently consumed by the Calvin-Benson cycle in the stroma to fix CO2 into G3P.",
     "high", "hard", "surface_matches_concept",
     "High-school / AP Biology (NGSS HS-LS1-5, HS-LS1-7). Surface matches."),

    ("Two plus two equals four.",
     "elementary", "easy", "surface_matches_concept",
     "Kindergarten arithmetic."),
    ("Solve for x: 3x + 7 = 22.",
     "middle", "easy", "surface_matches_concept",
     "Single-variable linear equation — middle-school algebra (Common Core 7.EE.B.4)."),
    ("Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
     "high", "medium", "surface_matches_concept",
     "Eigenvalue computation — grade 11-12 linear algebra preview / advanced precalculus."),

    ("The cat sat on the mat.",
     "elementary", "easy", "surface_matches_concept",
     "Classic K-1 reading sample."),
    ("Atoms are made of protons, neutrons, and electrons.",
     "middle", "easy", "surface_matches_concept",
     "Middle-school physical science."),
    ("Define enthalpy in the context of a closed thermodynamic system.",
     "high", "medium", "surface_matches_concept",
     "High-school AP Chemistry (NGSS HS-PS3-2)."),

    # More short prompts where the concept is doing the real work.
    ("What is gravity?",
     "elementary", "easy", "surface_matches_concept",
     "As a basic concept (things fall down), gravity is K-2. We deliberately use this as a sanity-check row — the concept is genuinely elementary at this level of phrasing."),
    ("What is the gravitational constant?",
     "high", "easy", "surface_easy_concept_hard",
     "G in Newton's universal law of gravitation — high-school physics (NGSS HS-PS2-4). Short question."),
    ("What causes earthquakes?",
     "middle", "easy", "surface_easy_concept_hard",
     "Plate tectonics / fault movement — middle school (NGSS MS-ESS2-2). Short."),
    ("Define oxidation.",
     "high", "easy", "surface_easy_concept_hard",
     "Oxidation-reduction reactions — high-school chemistry."),
    ("What is a fraction?",
     "elementary", "easy", "surface_matches_concept",
     "Fractions intro — grade 3 (Common Core 3.NF.A.1)."),
    ("Find the derivative of e to the x.",
     "high", "easy", "surface_easy_concept_hard",
     "Exponential function calculus — AP Calc."),
    ("What is osmosis?",
     "middle", "easy", "surface_easy_concept_hard",
     "Cellular transport — middle school (NGSS MS-LS1-2)."),
    ("Explain Newton's third law.",
     "middle", "easy", "surface_easy_concept_hard",
     "Newton's laws — middle-school physical science (NGSS MS-PS2-1)."),
    ("Convert 100 degrees Fahrenheit to Celsius.",
     "middle", "easy", "surface_matches_concept",
     "Temperature conversion — middle school math/science (typical grade 6-7)."),

    # Borderline cases and extra sanity checks.
    ("What is a sonnet?",
     "middle", "easy", "surface_easy_concept_hard",
     "Sonnet form — middle-school English language arts (Common Core 7-8 RL.5)."),
    ("Define mitochondria.",
     "middle", "easy", "surface_easy_concept_hard",
     "Cell organelles — middle-school life science (NGSS MS-LS1-2)."),
    ("Compute the volume of a sphere with radius 5.",
     "middle", "easy", "surface_easy_concept_hard",
     "Volume formula — middle school (Common Core 8.G.C.9)."),
    ("Apply L'Hopital's rule.",
     "high", "easy", "surface_easy_concept_hard",
     "L'Hopital's rule — AP Calculus."),
    ("Why is the sky blue?",
     "high", "easy", "surface_easy_concept_hard",
     "Rayleigh scattering explanation — high-school physics. The QUESTION sounds elementary but a curriculum-correct EXPLANATION belongs at high school."),
    ("What is a verb?",
     "elementary", "easy", "surface_matches_concept",
     "Parts of speech — grade 1-2 (Common Core L.1.1)."),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='build_adv_concept',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        out_path = os.path.join(DATA_DIR, 'adv_concept.csv')
        with open(out_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['idx', 'text', 'true_level', 'surface_complexity',
                        'category', 'reasoning', 'source'])
            for i, (text, true_lvl, surf, cat, why) in enumerate(SEED_ROWS):
                w.writerow([i, text, true_lvl, surf, cat, why,
                            'hand_curated_2026-05-23'])
        print(f"[build_adv_concept] wrote {_display_path(out_path)}")
        print(f"[build_adv_concept] {len(SEED_ROWS)} rows total")

        # Quick distribution check for accidental label/category drift.
        from collections import Counter
        cat_counts = Counter(r[3] for r in SEED_ROWS)
        lvl_counts = Counter(r[1] for r in SEED_ROWS)
        surf_counts = Counter(r[2] for r in SEED_ROWS)
        print(f"[build_adv_concept] by category:  {dict(cat_counts)}")
        print(f"[build_adv_concept] by true_level: {dict(lvl_counts)}")
        print(f"[build_adv_concept] by surface:    {dict(surf_counts)}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
