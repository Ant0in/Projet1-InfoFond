
import networkx as nx
import itertools

from pysat.solvers import Minicard
from pysat.formula import CNFPlus, IDPool


# Q2
def gen_solution(G: nx.Graph, k: int) -> list[tuple[int, set, set]]:
    
    """
    Génère une solution pour le problème d'Alcuin avec un Alcuin <= k.
    La solution respecte toutes les contraintes imposées (7).

    Args:
        G (nx.Graph): Graphe représentant les conflits entre les sujets.
        k (int): Limite supérieure du nombre de "steps" (alcuin(s) = alcuin(G))

    Returns:
        list[tuple[int, set, set]]: Liste des configurations valides sous la forme
                                    (boatman_side, left_side_subjects, right_side_subjects).
    """
    
    subjects: list = list(G.nodes)
    num_subjects: int = len(subjects)
    max_time_steps = (2 * num_subjects) + 2  # THM n°1
    
    pool: IDPool = IDPool()
    cnf: CNFPlus = CNFPlus()

    # - x[t, s, r]: Le sujet `s` est sur la rive `r` (0 = gauche, 1 = droite) au temps t.
    def var_x(t, s, r): return pool.id(f"x_{t}_{s}_{r}")
    # - b[t, r]: Le boatman est sur la rive `r` au temps t.
    def var_b(t, r): return pool.id(f"b_{t}_{r}")

    # 1st constraint : sequence must begins with configuration s0 (boatman left and subjects all left)
    cnf.append([var_b(0, 0)])
    cnf.append([-var_b(0, 1)])
    for s in range(num_subjects):
        cnf.append([var_x(0, s, 0)])
        cnf.append([-var_x(0, s, 1)])

    # 2nd constraint : sequence must ends with configuration sf (boatman right and subjects all right)
    cnf.append([var_b(max_time_steps-1, 1)])
    cnf.append([-var_b(max_time_steps-1, 0)])
    for s in range(num_subjects):
        cnf.append([var_x(max_time_steps-1, s, 1)])
        cnf.append([-var_x(max_time_steps-1, s, 0)])
        
    # 3rd constraint : only the subjects on the boatman side can switch side.
    for t in range(max_time_steps - 1):
        for s in range(num_subjects):
            # pour chaque ts, on check que :
            # si boatman left, alors ts+1 aucun des sujets right n'est left
            # si boatman right, alors ts+1 aucun des sujets left n'est right
            cnf.append([-var_b(t, 0), -var_x(t, s, 1), var_x(t+1, s, 1)])
            cnf.append([-var_b(t, 1), -var_x(t, s, 0), var_x(t+1, s, 0)])

    # 4th constraint : boatman and subjects are only on one side, not both, not None.
    for t in range(max_time_steps):
        cnf.append([var_b(t, 0), var_b(t, 1)])
        cnf.append([-var_b(t, 0), -var_b(t, 1)])
        for s in range(num_subjects):
            cnf.append([var_x(t, s, 0), var_x(t, s, 1)])
            cnf.append([-var_x(t, s, 0), -var_x(t, s, 1)])
            
    # 5th constraint : boatman switch sides each timestamp
    for t in range(max_time_steps - 1):
        cnf.append([-var_b(t, 0), var_b(t+1, 1)])
        cnf.append([-var_b(t, 1), var_b(t+1, 0)])

    # 6th constraint : no conflicts allowed on the side the boatman is not.
    for t in range(max_time_steps):
        for s1, s2 in list(itertools.combinations(subjects, 2)):
            if (s1, s2) in G.edges or (s2, s1) in G.edges:
                # conflict between s1 and s2. must not be on the same side if boatman is not here.
                cnf.append([-var_b(t, 0), -var_x(t, subjects.index(s1), 1), -var_x(t, subjects.index(s2), 1)])
                cnf.append([-var_b(t, 1), -var_x(t, subjects.index(s1), 0), -var_x(t, subjects.index(s2), 0)])

    # 7th constraint : from t -> t+1, the number of subject that moved (alcuin(c)) is inferior to k.
    # l'idée est de regarder entre deux instants si il existe un sous ensemble de taille k+1 de sujets qui ont bougé.
    # On va donc générer toutes les permutations de k+1 sujets.
    
    for subgroup in itertools.combinations(range(num_subjects), k+1):
        
        # Pour chaque subgroup, on regarde si ses éléments se sont tous déplacés.
        for t in range(max_time_steps - 1):
            for r in range(2):
                clause: list = []
                for i in subgroup:
                    clause.append(-var_x(t, i, r))
                    clause.append(-var_x(t + 1, i, (r + 1) % 2))
                cnf.append(clause)

    # solver :
    with Minicard(bootstrap_with=cnf) as solver:
        
        solution_found: bool = solver.solve()

        if solution_found:

            model: list | None = solver.get_model()
            solution: list[tuple[int, set, set]] = list()
            
            for t in range(max_time_steps):

                left_bank: set = set()
                right_bank: set = set()
                # boatman switches sides each step
                boatman_bank: int = 0 if t % 2 == 0 else 1
                
                for s in range(num_subjects):

                    left_index: int = var_x(t, s, 0) - 1
                    right_index: int = var_x(t, s, 1) - 1

                    if 0 <= left_index < len(model):
                        if model[left_index] > 0:
                            left_bank.add(subjects[s])

                    if 0 <= right_index < len(model):
                        if model[right_index] > 0:
                            right_bank.add(subjects[s])

                # if configuration is empty, we skip it.
                if len(left_bank) + len(right_bank) > 0:
                    solution.append((boatman_bank, left_bank, right_bank))

            return solution
        
        else:
            # No solution.
            return None
        
# Q3
def find_alcuin_number(G: nx.Graph) -> int:
    
    # key idea behind this function is that worst case alcuin for a graph G with n subject is n,
    # which means the boatman just brings everyone on his boat. We will try to find a solution
    # between 1 and n, and return the best one that comes first.

    n: int = len(G.nodes)

    for i in range(1, n + 1):

        solution: None | list[tuple[int, set, set]] = gen_solution(G=G, k=i)
        if solution is not None:
            return i
    
    # not supposed to have a alcuin > n. worst case scenario is n, for complete graph i.e.
    raise Exception('[E] Not supposed to reach this.')


# Q5
def gen_solution_cvalid(G: nx.Graph, k: int, c: int) -> list[tuple[int, set, set, tuple[set]]]:
    # TODO : À COMPLÉTER
    return

# Q6
def find_c_alcuin_number(G: nx.Graph, c: int) -> int:
    # TODO : À COMPLÉTER
    return




if __name__ == "__main__":

    G = nx.Graph()
    G.add_nodes_from(['choux', 'chèvre', 'loup'])
    G.add_edges_from([('choux', 'chèvre'), ('chèvre', 'loup')])
    solution: list[tuple[int, set, set]] = gen_solution(G, 1)

    print(solution)

