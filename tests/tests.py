
import sys
from pathlib import Path
# Pour permettre une structure propre, on doit ajouter le dossier ./src du projet
# dans le path des modules à importer.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from functools import wraps
from itertools import combinations
from inspect import getfullargspec

from src.utils import blue, red, green, magenta, INFINITY
from src.utils import (
    PathGraph, CompleteGraph, ClawGraph, EmptyGraph, PetersenGraph
)
from src.project import (
    gen_solution, find_alcuin_number,
    gen_solution_cvalid, find_c_alcuin_number
)

pos_success = 0
pos_fail    = 0
neg_success = 0
neg_fail    = 0

import atexit
def log_tests():
    tot = pos_success + pos_fail + neg_success + neg_fail
    if tot == 0:
        return
    print('')
    if neg_fail == 0:
        print(green('\tNo negative test failed!'))
    else:
        print(red(f'\t{neg_fail} negative tests failed!'))
    if pos_fail == 0:
        print(green('\tNo positive test failed!'))
    else:
        print(red(f'\t{pos_fail} positive tests failed!'))
atexit.register(log_tests)

def test_function(fct):
    @wraps(fct)
    def wrapper(*args, **kwargs):
        marker = blue('='*10)
        print(marker, 'Entering test function:', fct.__name__.ljust(15), marker)
        fct(*args, *kwargs)
        print(marker, 'Exiting test function: ', fct.__name__.ljust(15), marker, end='\n\n')
    return wrapper

def throw(msg, prefix=''):
    if prefix:
        msg = prefix + ' ' + msg
    raise ValueError(msg)

def verify(solution, G, k, c=None):
    vertices = set(G)
    if solution[0][1] != vertices:
        throw('Everybody should start on side 0')
    if solution[-1][2] != vertices:
        throw(' Everybody should end on side 1')
    for step, sol in enumerate(solution):
        b, S0, S1 = sol[:3]
        prefix = f'At time t={step}'

        def _throw(msg):
            raise ValueError(f'[{prefix}] {msg}')
        if b != (step&1):
            _throw(f'Invalid b={b}')
        if S0 | S1 != vertices or len(S0 & S1) > 0:
            _throw('Expected a partition')
        if step > 0:
            _, prevS0, prevS1 = solution[step-1][:3]
            diff0 = S0.symmetric_difference(prevS0)
            diff1 = S1.symmetric_difference(prevS1)
            assert diff0 == diff1
            X = diff0
            if len(X) > k:
                _throw(f'At most {k} movements allowed, not {len(X)}')
    if c is not None:
        for step, (b, S0, S1, Xps) in enumerate(solution):
            prefix = f'At time t={step}'

            def _throw(msg):
                raise ValueError(f'[{prefix}] {msg}')
            if step == 0:
                continue
            Xps = tuple(Xp for Xp in Xps if len(Xp) > 0)
            if len(Xps) > c:
                _throw(f'Expected at most {c} compartments, not {len(Xps)}')
            X = set.union(*Xps) if len(Xps) > 0 else set()
            diff = S0.symmetric_difference(solution[step-1][1])
            if X != diff:
                print((X, diff))
                _throw('The X_i\'s must coincide with the people who moved')
            for Xp in Xps:
                if any(G.has_edge(x, y) for x, y in combinations(Xp, 2)):
                    _throw('Cannot have conflict in the compartments')

def test_positive(caller, callback, instances, check_callback=verify):
    global pos_success, pos_fail
    nb_args = len(getfullargspec(callback).args)
    for counter, args in enumerate(instances, start=1):
        prefix = (f'[{caller}+:Test {counter}]').ljust(20)
        solution = callback(*args[:nb_args])
        if solution is None:
            print(f'\t{prefix} Test {red("failed")}: no solution found')
            pos_fail += 1
            continue
        try:
            check_callback(solution, *args)
        except ValueError as e:
            print(f'\t{prefix} Test {red("failed")}: {str(e)}')
            pos_fail += 1
        else:
            print(f'\t{prefix} Test {green("passed")}')
            pos_success += 1

def test_negative(caller, callback, instances):
    global neg_success, neg_fail
    nb_args = len(getfullargspec(callback).args)
    for counter, args in enumerate(instances, start=1):
        prefix = (f'[{caller}-:Test {counter}]').ljust(20)
        solution = callback(*args[:nb_args])
        if solution is None:
            print(f'\t{prefix} Test {green("passed")}')
            neg_success += 1
        else:
            print(f'\t{prefix} Test {red("failed")}')
            neg_fail += 1


@test_function
def test_basic_Q2():
    BASIC_INSTANCE = [
        (PathGraph(3), 1)
    ]
    test_positive('test_basic', gen_solution, BASIC_INSTANCE)

@test_function
def test_cliques_Q2():
    POSITIVE_INSTANCES = [
        (CompleteGraph(n), n-1)
        for n in range(3, 10)
    ]
    NEGATIVE_INSTANCES = [
        (CompleteGraph(n), n-2)
        for n in range(3, 10)
    ]
    test_positive('test_cliques', gen_solution, POSITIVE_INSTANCES)
    test_negative('test_cliques', gen_solution, NEGATIVE_INSTANCES)

@test_function
def test_claws_Q2():
    POSITIVE_INSTANCES = [
        (ClawGraph(n), 2)
        for n in range(3, 10)
    ]
    NEGATIVE_INSTANCES = [
        (ClawGraph(n), 1)
        for n in range(3, 10)
    ]
    test_positive('test_claws', gen_solution, POSITIVE_INSTANCES)
    test_negative('test_claws', gen_solution, NEGATIVE_INSTANCES)

def _verify_size(solution, G, k):
    if solution != k:
        throw(f'Expected {k} but got {solution}')

@test_function
def test_claws_Q3():
    POSITIVE_INSTANCES = [
        (ClawGraph(n), 2)
        for n in range(3, 10)
    ]  # + [(ClawGraph(2), 1)]

    test_positive('test_claws', find_alcuin_number, POSITIVE_INSTANCES, _verify_size)

@test_function
def test_cliques_Q3():
    POSITIVE_INSTANCES = [
        (CompleteGraph(n), n-1)
        for n in range(2, 10)
    ]
    POSITIVE_INSTANCES += [
        (EmptyGraph(1), 1)
    ]
    test_positive('test_cliques', find_alcuin_number, POSITIVE_INSTANCES, _verify_size)

@test_function
def test_petersen_Q3():
    POSITIVE_INSTANCES = [
        (PetersenGraph(), 6)  # tau(P) = Alcuin(P) = 6
    ]
    test_positive('test_petersen', find_alcuin_number, POSITIVE_INSTANCES, _verify_size)

@test_function
def test_basic_Q5():
    POSITIVE_INSTANCES = [
        (PathGraph(3), 2, 1),
        (PathGraph(3), 2, 2)
    ]
    NEGATIVE_INSTANCES = [
    ]
    test_positive('test_basic', gen_solution_cvalid, POSITIVE_INSTANCES)
    test_negative('test_basic', gen_solution_cvalid, NEGATIVE_INSTANCES)

@test_function
def test_cliques_Q5():
    POSITIVE_INSTANCES = [
        (CompleteGraph(n), n-1, n-1)
        for n in range(3, 10)
    ]
    NEGATIVE_INSTANCES = [
        (CompleteGraph(n), n-2, n-2)
        for n in range(3, 10)
    ] + [
        (CompleteGraph(n), n-1, n-2)
        for n in range(3, 10)
    ]
    test_positive('test_cliques', gen_solution_cvalid, POSITIVE_INSTANCES)
    test_negative('test_cliques', gen_solution_cvalid, NEGATIVE_INSTANCES)

@test_function
def test_claws_Q5():
    POSITIVE_INSTANCES = [
        (ClawGraph(n), (n+1)//2, 1)
        for n in range(3, 10)
    ] + [
        (ClawGraph(n), 2, 2)
        for n in range(3, 10)
    ]
    NEGATIVE_INSTANCES = [
        (ClawGraph(n), 1, n)
        for n in range(3, 10)
    ]
    test_positive('test_claws', gen_solution_cvalid, POSITIVE_INSTANCES)
    test_negative('test_claws', gen_solution_cvalid, NEGATIVE_INSTANCES)

def _verify_c_size(solution, G, c, k):
    if solution != k:
        throw(f'Expected {k} but got {solution}')

@test_function
def test_basic_Q6():
    POSITIVE_INSTANCES = [
        (PathGraph(3), 0, INFINITY),
        (PathGraph(3), 1, 1),
        (PathGraph(3), 2, 1)
    ]
    test_positive('test_basic', find_c_alcuin_number, POSITIVE_INSTANCES, _verify_c_size)

@test_function
def test_cliques_Q6():
    POSITIVE_INSTANCES = [
        (CompleteGraph(n), n-1, n-1)
        for n in range(3, 10)
    ] + [
        (CompleteGraph(n), n-2, INFINITY)
        for n in range(3, 10)
    ]
    test_positive('test_cliques', find_c_alcuin_number, POSITIVE_INSTANCES, _verify_c_size)

@test_function
def test_claws_Q6():
    POSITIVE_INSTANCES = [
        (ClawGraph(n), 1, (n+1)//2)
        for n in range(2, 10)
    ] + [
        (ClawGraph(n), 2, 2)
        for n in range(3, 10)
    ]
    test_positive('test_claws', find_c_alcuin_number, POSITIVE_INSTANCES, _verify_c_size)

@test_function
def test_petersen_Q6():
    POSITIVE_INSTANCES = [
        (PetersenGraph(), 1, INFINITY)
    ] + [
        (PetersenGraph(), n, 6)
        for n in range(2, 11)
    ]
    test_positive('test_petersen', find_c_alcuin_number, POSITIVE_INSTANCES, _verify_c_size)

def main():
    print(magenta('\n   ##########   Tests for Q2  ##########   \n'))
    test_basic_Q2()
    test_cliques_Q2()
    test_claws_Q2()
    print(magenta('\n   ##########   Tests for Q3  ##########   \n'))
    test_claws_Q3()
    test_cliques_Q3()
    test_petersen_Q3()
    print(magenta('\n   ##########   Tests for Q5  ##########   \n'))
    test_basic_Q5()
    test_cliques_Q5()
    test_claws_Q5()
    print(magenta('\n   ##########   Tests for Q6  ##########   \n'))
    test_basic_Q6()
    test_cliques_Q6()
    test_claws_Q6()
    test_petersen_Q6()

if __name__ == '__main__':
    main()
