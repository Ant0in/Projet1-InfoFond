import networkx as nx
import itertools


class AlcuinNumber:


    @staticmethod
    def generate_config(G: nx.Graph) -> any:
        # Générer toutes les configurations possibles du problème.
        comb = ((combo + (('boatman', b),)) for combo in itertools.product(*[[(x, i) for i in {0, 1}] for x in G.nodes]) for b in (0, 1))
        return comb

    @staticmethod
    def is_config_valid(g: nx.Graph, c: tuple[tuple]) -> bool:

        left: list = []; right: list = []
        edges = g.edges

        # On sépare les sujets en fonction de leur rive et on vérifie la présence du boatman
        for s, r in c:
            if r == 0: left.append(s)
            else: right.append(s)

        # On vérifie chaque rive, mais uniquement si le boatman n'est pas présent sur la rive
        for rive in [left, right]:
            if 'boatman' not in rive:
        
                for (s1, s2) in itertools.combinations(rive, 2):
                    # Si un sujet est en conflit avec un autre, et qu'ils sont sur la même rive, la config est invalide
                    if (s1, s2) in edges or (s2, s1) in edges:
                        return False
        return True

    @staticmethod
    def generate_valid_config(g: nx.Graph) -> list:
        # Génére toutes les configurations valides.
        valid_config: list = []
        for c in AlcuinNumber.generate_config(G=g):
            if AlcuinNumber.is_config_valid(g, c):
                valid_config.append(c)
        return valid_config

    @staticmethod
    def is_boatman_different(c1, c2) -> bool:
        # Vérifier si le boatman est sur une rive différente entre les deux configurations
        boatman1 = next(pos for item, pos in c1 if item == 'boatman')
        boatman2 = next(pos for item, pos in c2 if item == 'boatman')
        return boatman1 != boatman2

    @staticmethod
    def calculate_seq_alcuin(sequence: list) -> int:
        
        alcuin: int = 0
        
        # On check pour chaque paire
        for i in range(len(sequence) - 1):

            current: tuple = sequence[i]
            next_config: tuple = sequence[i + 1]
            
            subjects_in_boat: int = -1  # On ne compte pas le boatman
            for (s1, r1), (s2, r2) in zip(current, next_config):
                if s1 == s2 and r1 != r2:
                    subjects_in_boat += 1
            
            alcuin = max(alcuin, subjects_in_boat)

        return alcuin

    @staticmethod
    def find_valid_sequences(g: nx.Graph, configs: list, start: tuple, end: tuple) -> list:
        
        ret: list = []
        for seq in AlcuinNumber.find_sequences(configs, start, end):
            if AlcuinNumber.validate_sequence(seq, g):
                ret.append(seq)

        return ret

    @staticmethod
    def validate_sequence(sequence: list, g: nx.Graph) -> bool:

        # Vérifier qu'une séquence est valide : seuls les sujets du côté du boatman se déplacent entre chaque configuration
        
        for i in range(len(sequence) - 1):
            
            config = sequence[i]
            next_config = sequence[i + 1]

            boatman_side = next(pos for item, pos in config if item == 'boatman')
            next_boatman_side = 1 - boatman_side

            # On veut vérifier que seuls ceux du côté du boatman peuvent se déplacer.
            assert len(config) == len(next_config)
            for s in range(len(config)):

                if config[s][1] == next_boatman_side and next_config[s][1] == boatman_side:
                    return False
        
        return True

    @staticmethod
    def find_sequences(configs: list, start: tuple, end: tuple) -> list:
        
        # Trouve toutes les suites de configurations possibles du start à end
        visited: set = set()
        all_sequences: list = list()

        def dfs(path: list):
            current = path[-1]
            if current == end:
                all_sequences.append(path[:])
                return
            visited.add(current)

            for next_config in configs:
                # Si la prochaine config respecte bien b" = 1-b et n'est pas redondante dans s:
                if next_config not in visited and AlcuinNumber.is_boatman_different(current, next_config):
                    path.append(next_config)
                    dfs(path)
                    path.pop()
            visited.remove(current)

        dfs([start])
        return all_sequences

    @staticmethod
    def find_best_sequence(sequences: list) -> tuple:

        current_best: tuple | None = None
        current_best_alcuin: float = float('inf')
        current_best_len: float = float('inf')
    
        for seq in sequences:

            a: int = AlcuinNumber.calculate_seq_alcuin(sequence=seq)
            l: int = len(seq)

            if current_best is None:
                current_best = seq
                current_best_alcuin = a
                current_best_len = l
    
            else:

                if a < current_best_alcuin:
                    current_best = seq
                    current_best_alcuin = a
                    current_best_len = l

                elif a == current_best_alcuin:
                    if l <= current_best_len:
                        current_best = seq
                    current_best_alcuin = a
                    current_best_len = l

        return current_best

    @staticmethod
    def display_sequence(sequence: tuple) -> None:
        
        print(f"[S] Solution d'Alcuin : {AlcuinNumber.calculate_seq_alcuin(sequence)}")

        for c in sequence:
            print(c)



    @staticmethod
    def execute(graph: nx.Graph) -> None:

        # On génère en premier lieu toutes les configurations possibles VALIDES. (max (s+1)², pour s sujets.)
        configs: list[tuple] = AlcuinNumber.generate_valid_config(g=graph)

        # Ensuite, on récupère les éléments de début et les éléments de fin de la config.
        start: tuple = configs[0]
        end: tuple = configs[-1]

        # Puis on cherche les séquences possibles pour nos configurations,
        # c'est à dire celles qui débutent par [start] et terminent par [end].
        # Une fois nos séquences générées, on veut vérifier les séquences valides, c'est à dire
        # celles qui assurent que les seuls sujets pouvant se déplacer sont ceux étant du côté du boatman.
        valid_sequences: list = AlcuinNumber.find_valid_sequences(graph, configs, start, end)

        # Enfin, on veut trouver la séquence de plus petit Alcuin dans les séquences. On décide également de
        # préférer la séquence la plus courte en cas d'égalité.
        best_sequence: tuple = AlcuinNumber.find_best_sequence(sequences=valid_sequences)
        AlcuinNumber.display_sequence(sequence=best_sequence)




if __name__ == '__main__':

    G = nx.Graph()
    G.add_nodes_from(['choux', 'chèvre', 'loup'])                   # Sujets
    G.add_edges_from([('choux', 'chèvre'), ('loup', 'chèvre')])     # Contraintes | Conflits
    AlcuinNumber.execute(graph=G)


