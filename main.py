
from drafter import *
from dataclasses import dataclass, field
import re
from itertools import product
from math import gcd
from functools import reduce
from typing import Dict, List, Tuple


'''
ATOMIC WEIGHTS
'''

ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.003,
    "Li": 6.94, "Be": 9.012,
    "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999,
    "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078,
    "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933,
    "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Br": 79.904,
    "Ag": 107.868, "I": 126.904,
    "Ba": 137.327, "Pt": 195.084,
    "Au": 196.967, "Hg": 200.592,
    "Pb": 207.2
}


'''
CHEMICAL DATACLASS
'''

@dataclass
class Chemical:
    '''Stores the information and calculated values for one chemical in the reaction.'''
    formula: str
    is_reactant: bool
    molecular_weight: float
    starting_mass: float = 0.0
    moles_available: float = 0.0
    excess_remaining: float = 0.0
    coefficient: int = 1


'''
STATE DATACLASS
'''

@dataclass
class State:
    '''Stores the information shared between the website's pages while a calculation is in progress.'''
    reaction_equation: str = ""
    heat_of_reaction: float = 0.0
    heat_entered: bool = False

    product_masses: List[float] = field(default_factory=list)
    chemicals: List[Chemical] = field(default_factory=list)

    selected_product: str = ""
    harvested_amount: float = 0.0
    theoretical_yield: float = 0.0
    percent_yield: float = 0.0

    limiting_reactant: str = ""
    balanced_equation: str = ""

    error_message: str = ""


'''
HELPER 1: PARSE CHEMICAL FORMULAS
'''

def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a chemical formula into element counts.

    The nested parse_group function handles parentheses and subscripts.
    """
    """
    REGULAR EXPRESSION DOCUMENTATION:
    re.findall searches the formula for matching text patterns and returns
    each match as a token. [A-Z] means one uppercase letter; [a-z]?
    means an optional lowercase letter, so element symbols such as H
    and Na are recognized. \\d+ means one or more digits (the atom
    count), and [()] recognizes opening and closing parentheses.
    The next check confirms that the matches account for every character,
    so unsupported characters or malformed text are rejected.
    """
    tokens = re.findall(r'[A-Z][a-z]?|\d+|[()]', formula)

    if "".join(tokens) != formula or not tokens:
        raise ValueError("Invalid chemical formula: " + formula)

    position = 0

    def parse_group(expect_closing: bool = False) -> Dict[str, int]:
        """Parse one formula group and apply any group multiplier."""
        nonlocal position
        counts: Dict[str, int] = {}

        while position < len(tokens):
            token = tokens[position]

            if token == ")":
                if not expect_closing:
                    raise ValueError("Unmatched closing parenthesis.")
                position += 1
                return counts

            if token == "(":
                position += 1
                group = parse_group(True)
                if not group:
                    raise ValueError("Empty parentheses are not valid.")
                multiplier = 1

                if position < len(tokens) and tokens[position].isdigit():
                    multiplier = int(tokens[position])
                    if multiplier < 1:
                        raise ValueError("Subscripts must be positive.")
                    position += 1

                for element, amount in group.items():
                    counts[element] = (
                        counts.get(element, 0) + amount * multiplier
                    )

            elif token.isdigit():
                raise ValueError("Invalid chemical formula.")

            else:
                element = token
                position += 1

                if element not in ATOMIC_WEIGHTS:
                    raise ValueError("Unknown element: " + element)

                amount = 1
                if position < len(tokens) and tokens[position].isdigit():
                    amount = int(tokens[position])
                    if amount < 1:
                        raise ValueError("Subscripts must be positive.")
                    position += 1

                counts[element] = counts.get(element, 0) + amount

        if expect_closing:
            raise ValueError("Unmatched opening parenthesis.")
        return counts

    result = parse_group()
    if position != len(tokens) or not result:
        raise ValueError("Invalid chemical formula.")
    return result


'''
HELPER 2: MOLECULAR WEIGHT
'''

def molecular_weight(formula: str) -> float:

    '''Calculates a chemical's molar mass by adding the atomic weights of all its elements.'''
    atoms = parse_formula(formula)
    weight = 0.0

    for element, amount in atoms.items():

        weight += ATOMIC_WEIGHTS[element] * amount

    return weight


'''
HELPER 3: SPLIT CHEMICAL EQUATION
'''

def split_equation(equation: str) -> Tuple[List[str], List[str]]:

    """Split and validate reactant and product formulas."""
    equation = equation.replace("→", "->")
    equation = equation.replace("=", "->")

    if equation.count("->") != 1:

        raise ValueError(
            "Use exactly one arrow to separate "
            "reactants and products."
        )

    left, right = equation.split("->")

    def get_formulas(side: str) -> List[str]:

        '''Splits one side of an equation at plus signs and returns its validated chemical formulas.'''
        formulas = []

        for item in side.split("+"):

            item = item.strip()

            if not item:
                raise ValueError(
                    "Missing chemical formula."
                )

            '''
            Remove an optional coefficient.
            '''
            match = re.fullmatch(
                r'(\d+)\s*(.+)', item
            )

            if match:
                item = match.group(2).strip()

            parse_formula(item)
            formulas.append(item)

        return formulas

    reactants = get_formulas(left)
    products = get_formulas(right)

    if len(set(reactants)) != len(reactants):
        raise ValueError(
            "Do not enter duplicate reactants."
        )

    if len(set(products)) != len(products):
        raise ValueError(
            "Do not enter duplicate products."
        )

    if len(reactants) > 3:
        raise ValueError(
            "This calculator supports up to 3 reactants."
        )

    return reactants, products


'''
HELPER 4: BALANCE CHEMICAL EQUATION
'''

def balance_equation(
        reactants: List[str], products: List[str]
        ) -> Tuple[Dict[str, int], Dict[str, int]]:

    '''Finds whole-number coefficients that make the number of each element equal on both sides of the reaction.'''
    formulas = reactants + products

    if len(formulas) > 6:
        raise ValueError(
            "This calculator supports up to 6 "
            "different chemicals total."
        )

    atoms = []

    for formula in formulas:
        atoms.append(parse_formula(formula))

    elements = set()

    for atom_counts in atoms:
        elements.update(atom_counts.keys())

    best = None

    '''
    COEFFICIENT-SEARCH DOCUMENTATION:
    itertools.product creates every possible ordered combination of
    coefficients from 1 through 10, with one coefficient per chemical.
    For each combination, the loops calculate the total number of each
    element on the reactant side and product side. A combination is valid
    only when those totals match for every element. This is a brute-force
    search: it is straightforward to follow, but it can take longer as
    the number of chemicals increases. The search is limited to six
    chemicals and coefficients no larger than 10.
    Try coefficients from 1 through 10.
    '''
    for coefficients in product(
            range(1, 11), repeat=len(formulas)):

        valid = True

        for element in elements:

            left_total = 0
            right_total = 0

            for i in range(len(reactants)):

                left_total += (
                    coefficients[i]
                    * atoms[i].get(element, 0)
                )

            for i in range(
                    len(reactants), len(formulas)):

                right_total += (
                    coefficients[i]
                    * atoms[i].get(element, 0)
                )

            if left_total != right_total:

                valid = False
                break

        if valid:

            '''
            Divide all coefficients by their greatest common divisor
            (GCD). This reduces equivalent multiples, such as 2, 2, 2
            to the simplest ratio 1, 1, 1.
            '''
            common = reduce(gcd, coefficients)

            normalized = tuple(
                number // common
                for number in coefficients
            )

            if best is None or sum(normalized) < sum(best):
                best = normalized

    if best is None:

        raise ValueError(
            "Could not balance this equation. "
            "Check the formulas or try another equation."
        )

    reactant_coefficients = {}
    product_coefficients = {}

    for i in range(len(reactants)):

        reactant_coefficients[reactants[i]] = best[i]

    for i in range(len(products)):

        product_coefficients[products[i]] = (
            best[i + len(reactants)]
        )

    return reactant_coefficients, product_coefficients


'''
HELPER 5: DISPLAY BALANCED EQUATION
'''

def make_balanced_equation(
        reactants: List[str],
        products: List[str],
        reactant_coefficients: Dict[str, int],
        product_coefficients: Dict[str, int]) -> str:

    '''Builds a readable balanced equation using the formulas and their calculated coefficients.'''
    left = []
    right = []

    for formula in reactants:

        coefficient = reactant_coefficients[formula]

        if coefficient == 1:
            left.append(formula)

        else:
            left.append(
                str(coefficient) + " " + formula
            )

    for formula in products:

        coefficient = product_coefficients[formula]

        if coefficient == 1:
            right.append(formula)

        else:
            right.append(
                str(coefficient) + " " + formula
            )

    return (
        " + ".join(left)
        + " -> "
        + " + ".join(right)
    )



"""UNIT TESTING DOCUMENTATION:
The helper functions can be unit tested independently because they accept
inputs and return predictable calculation results. The direct assertions
below test the existing helper functions; they do not create separate test
functions or a unittest class.

The Drafter route/page functions are not unit tested here because they return
framework Page objects and depend on browser navigation, button clicks, form
submissions, and Drafter-managed state. Test those routes manually by running
the website and checking each page, its inputs, navigation, and displayed
results.
"""

# Unit tests for the existing helper functions.
assert parse_formula("H2O") == {"H": 2, "O": 1}
assert round(molecular_weight("H2O"), 3) == 18.015
assert split_equation("H2 + O2 -> H2O") == (
    ["H2", "O2"], ["H2O"]
)
assert balance_equation(["H2", "O2"], ["H2O"]) == (
    {"H2": 2, "O2": 1}, {"H2O": 2}
)
assert make_balanced_equation(
    ["H2", "O2"], ["H2O"],
    {"H2": 2, "O2": 1}, {"H2O": 2}
) == "2 H2 + O2 -> 2 H2O"



'''
ROUTE 1: HOME PAGE
'''

@route
def index(state: State) -> Page:

    '''Displays the home page and asks the user to enter a chemical equation.'''
    return Page(state, [

        Header("Chemical Reaction Calculator"),

        "Balance a chemical equation, then calculate "
        "its heat change or theoretical and percent yield.",

        "Enter an equation using + between chemicals "
        "and -> between reactants and products.",

        "Example: H2 + O2 -> H2O",

        Header("Chemical Equation"),

        TextBox(
            "equation",
            state.reaction_equation
        ),

        Button("Balance Equation", "pass_equation")
    ])


'''
ROUTE 2: PROCESS EQUATION
'''

@route
def pass_equation(
        state: State,
        equation: str) -> Page:

    """Balance the submitted equation and save its chemicals in State."""
    try:

        if not equation.strip():

            raise ValueError(
                "Enter a chemical equation."
            )

        reactants, products = split_equation(equation)

        if len(reactants) > 3:

            raise ValueError(
                "This calculator supports up to 3 reactants."
            )

        reactant_coefficients, product_coefficients = (
            balance_equation(reactants, products)
        )

        '''
        Save the new equation.
        '''
        state.reaction_equation = equation

        state.heat_of_reaction = 0.0
        state.heat_entered = False

        state.chemicals = []
        state.product_masses = []

        state.selected_product = ""
        state.harvested_amount = 0.0
        state.theoretical_yield = 0.0
        state.percent_yield = 0.0
        state.limiting_reactant = ""

        '''
        Create reactant objects.
        '''
        for formula in reactants:

            chemical = Chemical(
                formula=formula,
                is_reactant=True,
                molecular_weight=molecular_weight(formula),
                coefficient=reactant_coefficients[formula]
            )

            state.chemicals.append(chemical)

        '''
        Create product objects.
        '''
        for formula in products:

            chemical = Chemical(
                formula=formula,
                is_reactant=False,
                molecular_weight=molecular_weight(formula),
                coefficient=product_coefficients[formula]
            )

            state.chemicals.append(chemical)

        state.balanced_equation = make_balanced_equation(
            reactants,
            products,
            reactant_coefficients,
            product_coefficients
        )

        return report_choices(state)

    except Exception as error:

        state.error_message = str(error)
        return error_page(state)


'''
ROUTE 3: REPORT SELECTION
'''

@route
def report_choices(state: State) -> Page:

    '''Displays the balanced equation and lets the user choose thermodynamic or stoichiometric analysis.'''
    return Page(state, [

        Header("Equation Balanced"),

        "Your balanced equation is:",

        state.balanced_equation,

        "Choose the type of analysis you want to run:",

        Button(
            "Thermodynamic Analysis",
            "thermo_input"
        ),

        Button(
            "Stoichiometric Yield",
            "input_masses"
        ),

        Button("Return Home", "index")
    ])


'''
ROUTE 4: THERMODYNAMIC INPUT
'''

@route
def thermo_input(state: State) -> Page:

    '''Displays the heat-of-reaction input page and preserves a previously entered heat value.'''
    if state.heat_entered:

        heat_text = str(state.heat_of_reaction)

    else:

        heat_text = ""

    return Page(state, [

        Header("Enter Heat of Reaction"),

        "Reaction:",

        state.balanced_equation,

        "Enter the heat change in kilojoules (kJ).",

        "Use a negative value when heat is released "
        "and a positive value when heat is absorbed.",

        "Heat Change (kJ):",

        TextBox("heat_value", heat_text),

        Button(
            "Calculate Thermodynamics",
            "calculate_thermo_action"
        )
    ])


'''
ROUTE 5: SAVE HEAT AND CALCULATE
'''

@route
def calculate_thermo_action(
        state: State,
        heat_value: str) -> Page:

    '''Checks and saves the entered heat value, then opens the thermodynamic report.'''
    try:

        if not heat_value.strip():

            raise ValueError(
                "Please enter the heat of reaction."
            )

        state.heat_of_reaction = float(heat_value)

        state.heat_entered = True

        return view_thermo_report(state)

    except ValueError:

        state.error_message = (
            "Please enter a valid numerical heat value."
        )

        return error_page(state)


'''
ROUTE 6: THERMODYNAMIC REPORT
'''

@route
def view_thermo_report(state: State) -> Page:

    '''Classifies the reaction as exothermic, endothermic, or thermoneutral and displays the explanation.'''

    '''
    If heat has not been entered, request it.
    '''
    if not state.heat_entered:

        return thermo_input(state)

    if state.heat_of_reaction < 0:

        reaction_type = "Exothermic"

        explanation = (
            "The reaction releases heat "
            "into its surroundings."
        )

    elif state.heat_of_reaction > 0:

        reaction_type = "Endothermic"

        explanation = (
            "The reaction absorbs heat "
            "from its surroundings."
        )

    else:

        reaction_type = "Thermoneutral"

        explanation = (
            "The reaction has no net heat change."
        )

    return Page(state, [

        Header("Thermodynamic Results"),

        "Reaction: " + state.balanced_equation,

        "Heat Change: "
        + str(state.heat_of_reaction) + " kJ",

        "Classification: " + reaction_type,

        explanation,

        Button(
            "Change Heat of Reaction",
            "thermo_input"
        ),

        Button(
            "Choose Your Calculation",
            "report_choices"
        ),

        Button("Start Over", "reset_calculation_action")
    ])


'''
ROUTE 7: REACTANT MASS INPUT
'''

@route
def input_masses(state: State) -> Page:

    '''Displays a separate mass input for each of up to three reactants and lets the user select a target product.'''
    reactants = []
    products = []

    for chemical in state.chemicals:

        if chemical.is_reactant:
            reactants.append(chemical)

        else:
            products.append(chemical)

    if len(reactants) > 3:

        state.error_message = (
            "This calculator supports up to 3 reactants."
        )

        return error_page(state)

    contents = [

        Header("Enter Reactant Masses"),

        "Reaction: " + state.balanced_equation,

        "Enter the amount you have of each reactant.",

        "Enter every mass in grams (g). Then choose "
        "which product you want to calculate."
    ]

    '''
    Create only the input boxes needed for this reaction.
    '''
    for i in range(len(reactants)):

        chemical = reactants[i]

        label = (
            "Reactant " + str(i + 1)
            + ": " + chemical.formula
        )

        '''
        Preserve previously entered masses.
        '''
        if chemical.starting_mass > 0:

            default_mass = str(
                chemical.starting_mass
            )

        else:

            default_mass = ""

        contents.append(Header(label))

        contents.append("Starting Mass (g):")

        contents.append(
            TextBox(
                "mass" + str(i + 1),
                default_mass
            )
        )

    product_names = []

    for chemical in products:

        product_names.append(chemical.formula)

    contents.extend([

        Header("Choose Product"),

        "Which product should the calculator find the "
        "theoretical yield for?",

        SelectBox(
            "target_product",
            product_names
        ),

        Button(
            "Calculate Theoretical Yield",
            "calculate_stoichiometry_action"
        )
    ])

    return Page(state, contents)


'''
ROUTE 8: STOICHIOMETRY CALCULATIONS
'''

@route
def calculate_stoichiometry_action(
        state: State,
        mass1: str = "",
        mass2: str = "",
        mass3: str = "",
        target_product: str = "") -> Page:

    """Validate masses and calculate limiting reactant and product yield."""
    try:

        reactants = []

        for chemical in state.chemicals:

            if chemical.is_reactant:
                reactants.append(chemical)

        if len(reactants) > 3:

            raise ValueError(
                "This calculator supports up to 3 reactants."
            )

        entered_masses = [mass1, mass2, mass3]
        masses = []

        '''
        Validate each required mass.
        '''
        for i in range(len(reactants)):

            if not entered_masses[i].strip():

                raise ValueError(
                    "Please enter a mass for "
                    + reactants[i].formula + "."
                )

            try:

                mass = float(entered_masses[i])

            except ValueError:

                raise ValueError(
                    "Enter a valid number for "
                    + reactants[i].formula + "."
                )

            if mass <= 0:

                raise ValueError(
                    "The mass of "
                    + reactants[i].formula
                    + " must be greater than zero."
                )

            masses.append(mass)

        '''
        Calculate moles available.
        '''
        for i in range(len(reactants)):

            reactants[i].starting_mass = masses[i]

            reactants[i].moles_available = (
                masses[i] / reactants[i].molecular_weight
            )

        '''
        Find the limiting reactant.
        '''
        ratios = []

        for chemical in reactants:

            ratios.append(
                chemical.moles_available
                / chemical.coefficient
            )

        limiting_amount = min(ratios)

        limiting_index = ratios.index(limiting_amount)

        state.limiting_reactant = (
            reactants[limiting_index].formula
        )

        '''
        Calculate excess remaining.
        '''
        for chemical in reactants:

            consumed = (
                limiting_amount * chemical.coefficient
            )

            chemical.excess_remaining = (
                chemical.moles_available - consumed
            )

            if abs(chemical.excess_remaining) < 0.000000001:

                chemical.excess_remaining = 0.0

        '''
        Calculate theoretical product masses.
        '''
        state.product_masses = []
        state.selected_product = target_product
        state.theoretical_yield = 0.0

        for chemical in state.chemicals:

            if not chemical.is_reactant:

                product_moles = (
                    limiting_amount * chemical.coefficient
                )

                product_mass = (
                    product_moles * chemical.molecular_weight
                )

                state.product_masses.append(product_mass)

                if chemical.formula == target_product:

                    state.theoretical_yield = product_mass

        if state.theoretical_yield <= 0:

            raise ValueError(
                "Please select a valid product."
            )

        '''
        Reset previous harvested mass results
        because the reactant masses may have changed.
        '''
        state.harvested_amount = 0.0
        state.percent_yield = 0.0

        return stoichiometry_report(state)

    except Exception as error:

        state.error_message = str(error)
        return error_page(state)


'''
ROUTE 9: STOICHIOMETRIC REPORT
'''

@route
def stoichiometry_report(state: State) -> Page:

    """Display limiting reactant, theoretical yield, and mass details."""
    contents = [

        Header("Theoretical Yield Results"),

        "Reaction: " + state.balanced_equation,

        "Limiting Reactant: " + state.limiting_reactant,

        "Product: " + state.selected_product,

        "Maximum Possible Product (Theoretical Yield): "
        + str(round(state.theoretical_yield, 4))
        + " g",

        Header("Reactant Information")
    ]

    '''
    Display information for every reactant.
    '''
    for chemical in state.chemicals:

        if chemical.is_reactant:

            contents.append(
                chemical.formula
                + " - Starting Mass: "
                + str(round(chemical.starting_mass, 4))
                + " g, Excess Remaining: "
                + str(round(chemical.excess_remaining, 4))
                + " mol"
            )

    '''
    Display previous percent yield if available.
    '''
    if state.harvested_amount > 0:

        contents.append(
            "Harvested Mass: "
            + str(round(state.harvested_amount, 4))
            + " g"
        )

        contents.append(
            "Percent Yield: "
            + str(round(state.percent_yield, 2))
            + "%"
        )

    contents.extend([

        "Next, enter how much product you actually collected "
        "to calculate percent yield (g):",

        TextBox(
            "harvested_mass",
            str(state.harvested_amount)
            if state.harvested_amount > 0 else ""
        ),

        Button(
            "Calculate Percent Yield",
            "view_yield_report"
        )
    ])

    return Page(state, contents)


'''
ROUTE 10: PERCENT YIELD REPORT
'''

@route
def view_yield_report(
        state: State,
        harvested_mass: str) -> Page:

    """Calculate percent yield and display the results and any warning."""
    try:

        if not harvested_mass.strip():

            raise ValueError(
                "Please enter the harvested product mass."
            )

        harvested = float(harvested_mass)

        if harvested < 0:

            raise ValueError(
                "Harvested mass cannot be negative."
            )

        if state.theoretical_yield <= 0:

            raise ValueError(
                "Theoretical yield must be greater than zero."
            )

        state.harvested_amount = harvested

        state.percent_yield = (
            harvested / state.theoretical_yield
        ) * 100

        report = [

            Header("Percent Yield Results"),

            "Reaction: " + state.balanced_equation,

            "Limiting Reactant: " + state.limiting_reactant,

            "Product: " + state.selected_product,

            "Theoretical Yield: "
            + str(round(state.theoretical_yield, 4))
            + " g",

            "Harvested Mass: "
            + str(round(state.harvested_amount, 4))
            + " g",

            "Percent Yield: "
            + str(round(state.percent_yield, 2))
            + "%"
        ]

        '''
        Check whether harvested mass exceeds
        theoretical yield.
        '''
        if state.harvested_amount > state.theoretical_yield:

            report.append(
                "Check your measurements: the collected mass "
                "is greater than the theoretical maximum."
            )

        report.extend([

            Button(
                "Choose Your Calculation",
                "report_choices"
            ),

            Button(
                "Start Over",
                "reset_calculation_action"
            )
        ])

        return Page(state, report)

    except Exception as error:

        state.error_message = str(error)
        return error_page(state)


'''
ROUTE 11: ERROR / VALIDATION PAGE
'''

@route
def error_page(state: State) -> Page:

    '''Displays validation or calculation errors and provides navigation back to the calculator.'''
    return Page(state, [

        Header("Please Check Your Entry"),

        state.error_message,

        "Correct the information and try again, or return "
        "to the calculation choices.",

        Button(
            "Return to Calculation Choices",
            "report_choices"
        ),

        Button(
            "Return Home",
            "index"
        ),

        Button(
            "Start Over",
            "reset_calculation_action"
        )
    ])


'''
ROUTE 12: RESET CALCULATION
'''

@route
def reset_calculation_action(state: State) -> Page:

    '''Clears the current reaction and all saved calculation results, then returns to the home page.'''
    state.reaction_equation = ""

    state.heat_of_reaction = 0.0
    state.heat_entered = False

    state.product_masses = []
    state.chemicals = []

    state.selected_product = ""

    state.harvested_amount = 0.0
    state.theoretical_yield = 0.0
    state.percent_yield = 0.0

    state.limiting_reactant = ""
    state.balanced_equation = ""

    state.error_message = ""

    return index(state)


start_server(State())



