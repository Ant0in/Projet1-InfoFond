
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


class cPartitionHelper:
    
    @staticmethod
    def generate_c_partitions(subjects: list[int], c: int) -> ...:

        n: int = len(subjects)
        
        # each element of s is assigned to a subset
        for assignment in itertools.product(range(c), repeat=n):
            # create subsets
            subsets: list[set] = [set() for _ in range(c)]
            for i, subset_index in enumerate(assignment):
                subsets[subset_index].add(subjects[i])
            # yield current subset
            yield subsets

    @staticmethod
    def find_stable_c_partition(G: nx.graph, c_partitions: list[list[set]]) -> list[set] | None:

        # Pour chaque partition dans les c_partitions, on va vérifier si elle est stable.

        for partition in c_partitions:
            
            is_partition_stable: bool = True

            # On vérifie chaque compartiment, pour vérifier qu'il est stable.
            for compartiment in partition:

                # Si le compartiment est occupé par un seul ou aucun sujet, alors aucun conflit.
                if len(compartiment) < 2: continue

                # Sinon on vérifie pour chaque paire de sujets qu'ils ne sont pas en conflit.
                else:

                    for s1, s2 in list(itertools.combinations(compartiment, 2)):

                        if (s1, s2) in G.edges or (s2, s1) in G.edges:
                            # La partition n'est pas stable puiqu'il existe une
                            # paire problématique dans la partition
                            is_partition_stable = False
                            break
                
                # Si un compartiment n'est pas stable, alors pas la peine de vérifier les autres.
                if not is_partition_stable:
                    break
            
            if is_partition_stable:
                # Si après avoir inspecté la partition on définit qu'elle est stable,
                # alors ça signifie qu'il existe un agencement des sujets tel que
                # les sujets fit sans conflit dans les c compartiments du bateau.
                return partition

        # On return une partition stable si elle existe en c éléments.
        # Il pourrait en exister plus, mais on ne les cherche pas, savoir qu'il existe
        # une partition stable (ou pas) est suffisant.
        return None


# Q5
def gen_solution_cvalid(G: nx.Graph, k: int, c: int) -> list[tuple[int, set, set, tuple[set]]]:
    
    """
    Génère une solution pour le problème d'Alcuin avec un Alcuin <= k et c compartiments de taille c.
    La solution respecte toutes les contraintes imposées.

    Args:
        G (nx.Graph): Graphe représentant les conflits entre les sujets.
        k (int): Limite supérieure du nombre de "steps" (alcuin(s) = alcuin(G))
        c (int): Nombre de compartiments.

    Returns:
        list[tuple[int, set, set, set]]: Liste des configurations valides sous la forme
            (boatman_side, left_side_subjects, right_side_subjects, boat_subjects (# = c)).
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

    # 8th constraint : from t -> t+1, the subject that moved (alcuin(c)) must be able to split into c groups
    # which dont create conflicts on the boat. Intuitively, we want to generate each c-subdivision of the set
    # of the subjects that are moving, and see if any of them is without conflicts. If not the case, then must be invalid transition.
    
    partition_dict: dict = dict()

    # Pour chaque sous groupe possible de sujets, on va regarder si une de ses c-partition est stable.
    for subgroup in frozenset(itertools.chain.from_iterable(itertools.combinations(subjects, r) for r in range(1, num_subjects + 1))):
        
        c_partitions: list[set] = list(cPartitionHelper.generate_c_partitions(subjects=subgroup, c=c))
        stable_c_partition: list[set] | None = cPartitionHelper.find_stable_c_partition(G=G, c_partitions=c_partitions)

        # Si une c-partition stable existe, alors on ne posera pas de contrainte sur ce transport.
        # Si elle ne l'est pas, alors on pose la contrainte que le transport de tout les sujets
        # (en même temps) du "subgroup" n'est pas possible avec c compartiments.

        if stable_c_partition is None:

            # print(f'[!] Subgroup problématique avec {c} compartiments : {subgroup}')

            for t in range(max_time_steps - 1):
                # TODO : fix the constraint
                for r in range(2):
                    clause: list = []
                    for i in subgroup:
                        clause.append(-var_x(t, i, r))
                        clause.append(-var_x(t + 1, i, (r + 1) % 2))

        else:
            partition_dict[frozenset(subgroup)] = stable_c_partition

    with Minicard(bootstrap_with=cnf) as solver:
        solution_found: bool = solver.solve()

        if solution_found:

            model: list | None = solver.get_model()
            solution: list[tuple[int, set, set]] = []
            empty_boat_c_partition: tuple[set] = tuple(set() for _ in range(c))
            
            for t in range(max_time_steps):
                # Initialisation des rives
                left_bank: set = set()
                right_bank: set = set()
                boat_subjects: set = None            
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

                if t != 0:

                    if boatman_bank == 0:
                        # le déplacement précédent était droite vers gauche
                        boat_subjects = partition_dict.get(frozenset(left_bank - solution[-1][1]))
                    else:
                        # le déplacement précédent était gauche vers droite
                        boat_subjects = partition_dict.get(frozenset(right_bank - solution[-1][2]))

                if boat_subjects is None:
                    boat_subjects = empty_boat_c_partition
                else:
                    boat_subjects = tuple(i for i in boat_subjects)

                solution.append((boatman_bank, left_bank, right_bank, boat_subjects))

            print(solution)
            return solution
        
        else:
            # No solution.
            return None

# Q6
def find_c_alcuin_number(G: nx.Graph, c: int) -> int | float:
    
    # key idea behind this function is that worst case c-alcuin for a graph G with n subject is n,
    # which means the boatman just brings everyone on his boat with n cells. We will try to find a solution
    # between 1 and c for c_prime, and for this c-Alcuin, a solution bewteen 1 and n, and return the best one that comes first.

    n: int = len(G.nodes)

    for k in range(1, n + 1):

        solution: None | list[tuple[int, set, set, set]] = gen_solution_cvalid(G=G, k=k, c=c)
        if solution is not None:
            return k
    
    # If found nothing, then returns +inf
    return float('inf')



if __name__ == '__main__':
    
    G = nx.Graph()
    G.add_nodes_from(['chevre', 'choux', 'loup'])
    G.add_edges_from([('chevre', 'loup'), ('chevre', 'choux')])
    print(find_c_alcuin_number(G=G, c=2))
