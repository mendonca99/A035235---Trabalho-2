# Implementação de uma árvore de decisão para classificação de dados.
import pandas as pd
from collections import Counter
from math import log2

#A classe DecisionTreeNode define um nó de uma árvore de decisão, armazenando atributos, filhos e previsões.
class DecisionTreeNode:
    def __init__(self):
        self.attribute = None
        self.children = {}
        self.leaf = False
        self.prediction = None
        self.samples = 0
        self.class_distribution = None

class DecisionTree:
    #Inicializa a árvore com raiz e profundidade máxima opcional.
    def __init__(self, max_depth=None):
        self.root = DecisionTreeNode()
        self.max_depth = max_depth

    #Calcula a entropia de um conjunto de rótulos.
    def calculate_entropy(self, labels):
        if len(labels) == 0:
            return 0
        counter = Counter(labels)
        entropy = 0
        total = len(labels)
        for count in counter.values():
            prob = count / total
            entropy -= prob * log2(prob)
        return entropy

    #Calcula o ganho de informação de um atributo em relação ao rótulo.
    def calculate_information_gain(self, data, attribute, target):
        total_entropy = self.calculate_entropy(data[target])
        weighted_entropy = 0

        # Calcular a entropia ponderada para cada valor do atributo
        for value in data[attribute].dropna().unique():
            subset = data[data[attribute] == value]
            weight = len(subset) / len(data)
            weighted_entropy += weight * self.calculate_entropy(subset[target])

        return total_entropy - weighted_entropy


    #Encontra o melhor atributo para dividir o conjunto de dados.
    def find_best_split(self, data, attributes, target):
        best_gain = -1
        best_attribute = None

        for attribute in attributes:
            gain = self.calculate_information_gain(data, attribute, target)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute

        return best_attribute

    #Constrói a árvore de decisão recursivamente.
    def build_tree(self, data, attributes, target, depth=0):
        node = DecisionTreeNode()
        node.samples = len(data)
        node.class_distribution = dict(Counter(data[target]))

        # Verificar se todos os exemplos possuem o mesmo rótulo
        if len(data[target].unique()) == 1:
            node.leaf = True
            node.prediction = data[target].iloc[0]
            return node

        # Verificar se não há mais atributos para dividir ou se atingiu a profundidade máxima
        if not attributes or (self.max_depth and depth >= self.max_depth):
            node.leaf = True
            node.prediction = data[target].mode().iloc[0]
            return node

        # Encontrar o melhor atributo para dividir
        best_attribute = self.find_best_split(data, attributes, target)
        node.attribute = best_attribute

        # Remover o melhor atributo da lista de atributos restantes
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # Construir os filhos do nó atual
        for value in data[best_attribute].dropna().unique():
            subset = data[data[best_attribute] == value]
            if subset.empty:
                child_node = DecisionTreeNode()
                child_node.leaf = True
                child_node.prediction = data[target].mode().iloc[0]
                child_node.samples = 0
                node.children[value] = child_node
            else:
                node.children[value] = self.build_tree(subset, remaining_attributes, target, depth + 1)

        # Tratar valores ausentes (NaN)
        nan_subset = data[data[best_attribute].isna()]
        if not nan_subset.empty:
            child_node = DecisionTreeNode()
            child_node.leaf = True
            child_node.prediction = data[target].mode().iloc[0]
            child_node.samples = len(nan_subset)
            child_node.class_distribution = dict(Counter(nan_subset[target]))
            node.children["NaN"] = child_node

        return node

    #Treina a árvore de decisão com os dados fornecidos.
    def fit(self, data, target):
        self.target_column = target
        attributes = [col for col in data.columns if col != target and col != "ID"]
        self.root = self.build_tree(data, attributes, target)

    #Faz uma previsão para um exemplo de entrada.
    def print_tree_formatted(self, node=None, indent="", level=0):
        if node is None:
            node = self.root

        if node.leaf:
            correct = node.class_distribution.get(node.prediction, 0) if node.class_distribution else 0
            total = node.samples
            errors = total - correct
            print(f"{indent}<leaf class=\"{node.prediction}\" count=\"{total}\" correct=\"{correct}\" errors=\"{errors}\" />")
        else:
            print(f"{indent}<attribute name=\"{node.attribute}\">")
            for value, child in node.children.items():
                print(f"{indent}  <value name=\"{value}\">")
                self.print_tree_formatted(child, indent + "    ", level + 1)
                print(f"{indent}  </value>")
            print(f"{indent}</attribute>")

    #Imprime os caminhos da árvore de decisão.
    def print_tree_paths(self, node=None, path="Root"):
        if node is None:
            node = self.root

        # Verificar se é um nó folha ou não
        if node.leaf:
            correct = node.class_distribution.get(node.prediction, 0) if node.class_distribution else 0
            total = node.samples
            errors = total - correct
            print(f"{path} -> Predict: {node.prediction} (Samples: {total}, Correct: {correct}, Errors: {errors})")
        else:
            for value, child in node.children.items():
                if child is None:
                    continue  # Pula nós não inicializados
                new_path = f"{path} -> {node.attribute}={value}"
                self.print_tree_paths(child, new_path)

# Carregar o dataset
restaurant_df = pd.read_csv('restaurant.csv')

# Construir a árvore e imprimir nos dois formatos
tree = DecisionTree(max_depth=5)
tree.fit(restaurant_df, "Class")

# Imprimir a árvore em formato XML - Figura 4
print("<decision_tree>")
tree.print_tree_formatted()
print("</decision_tree>")

# Imprimir os caminhos da árvore  - Figura 5 
print("\nPaths for each decision:")
tree.print_tree_paths()
