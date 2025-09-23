import random
from dataclasses import dataclass


@dataclass
class ArithmeticProblem:
    num_1: int
    num_2: int
    num_3: int
    num_4: int
    result: int


class ArithmeticProblemGenerator:
    def __init__(
        self,
        min_num: int = 1,
        max_num: int = 100,
        result_min: int = 1,
        result_max: int = 1000,
        max_attempts: int = 100,
        operators: tuple[str] = ("+", "-", "*", "/"),
    ):
        """
        Initialize the arithmetic problem generator.

        Args:
            min_num: The minimum number to use in the arithmetic problem
            max_num: The maximum number to use in the arithmetic problem
            operators: The operators to use in the arithmetic problem
        """
        self.min_num = min_num
        self.max_num = max_num
        self.result_min = result_min
        self.result_max = result_max
        self.operators = operators
        self.max_attempts = max_attempts

    def _generate_random_number(self) -> int:
        return random.randint(self.min_num, self.max_num)

    def _generate_random_operator(self) -> str:
        return random.choice(self.operators)

    def generate_problem(self) -> ArithmeticProblem:
        """
        Generate an countdown arithmetic problem.

        Generate four numbers, num_1, num_2, num_3, num_4,
        and operators between them, and apply the operators to the numbers to get the result.

        Make sure that the result must exactly be an integer, and
        match the result of the arithmetic problem.

        Returns:
            ArithmeticProblem: The generated arithmetic problem
        """
        max_attempts = 100

        for _ in range(max_attempts):
            # Generate four random numbers
            num_1 = self._generate_random_number()
            num_2 = self._generate_random_number()
            num_3 = self._generate_random_number()
            num_4 = self._generate_random_number()

            # Generate three random operators for the expression: num_1 op1 num_2 op2 num_3 op3 num_4
            op1 = self._generate_random_operator()
            op2 = self._generate_random_operator()
            op3 = self._generate_random_operator()

            # Try to evaluate the expression and ensure it results in an integer
            result = self._evaluate_expression(
                num_1, op1, num_2, op2, num_3, op3, num_4
            )

            # Check if result is an integer (no floating point remainder)
            if (
                isinstance(result, (int, float))
                and result == int(result)
                and self.result_min <= result <= self.result_max
            ):
                result = int(result)
                return ArithmeticProblem(
                    num_1=num_1, num_2=num_2, num_3=num_3, num_4=num_4, result=result
                )

        return None

    def _evaluate_expression(
        self,
        num_1: int,
        op1: str,
        num_2: int,
        op2: str,
        num_3: int,
        op3: str,
        num_4: int,
    ) -> float:
        """
        Evaluate the arithmetic expression following standard order of operations.

        Args:
            num_1: First number
            op1: First operator
            num_2: Second number
            op2: Second operator
            num_3: Third number
            op3: Third operator
            num_4: Fourth number

        Returns:
            float: The result of the arithmetic expression
        """
        # Build expression string: num_1 op1 num_2 op2 num_3 op3 num_4
        expression = f"{num_1} {op1} {num_2} {op2} {num_3} {op3} {num_4}"

        # Use eval to calculate the result (following Python's order of operations)
        # This handles operator precedence correctly (* and / before + and -)
        return eval(expression)
