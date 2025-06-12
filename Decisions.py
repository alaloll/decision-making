import sys
import numpy as np
from itertools import product
from collections import deque

#вспомогательные функции
def n_input(s):
    while True:
        try:
            return int(input(s))
        except ValueError:
            print("Ошибка: ожидается целое число. Попробуйте снова.\n")

def scale():
    num=input("Введите диапазон шкалы (например, 1-5): ").split("-")
    if len(num)!= 2:
        print("Нужно указать две границы через пробел.")
        return scale()
    l,h=map(int, num)
    if l>=h:
        print("Нижняя граница должна быть меньше верхней.")
        return scale()
    return l, h

#парето 
def pareto(alts):
    A = np.array(alts)
    d=[0]*len(A)
    l=len(A[0])
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            cnt1=0
            cnt2=0
            for k in range(l):
                if A[j][k]>=A[i][k]:
                    cnt1+=1
                if A[i][k]>=A[j][k]:
                    cnt2+=1
            if cnt1==l:
                d[i]=1
            if cnt2==l:
                d[j]=1
    for i in range(len(A)):
        if d[i]==0:
            d[i]=1
        else:
            d[i]=0
    return [j for j in range(len(alts)) if d[j]==1]

#создаём граф
class Graph:
    def __init__(self):
        self.edges = {}
    def add_node(self,u):
        self.edges.setdefault(u, [])
    def add_edge(self,u,v):
        self.edges.setdefault(u, [])
        self.edges.setdefault(v, [])
        self.edges[u].append(v)
    def neighbors(self, node):
        return self.edges.get(node, [])
    def nodes(self):
        return list(self.edges)
    
#качественный этап
def add_pareto_edges(G: Graph, m, l):
    """Добавляем рёбра Парето в наш граф. Достаточно просто 
    для каждой вершины добавить рёбра к её "соседям снизу" -
    по транзитивности можно восстановить полную информацию
    доминирования вершин друг над другом по Парето.

    Аргументы:
        G : граф критериального пространства (ГКП)
        m : число критериев
        l : минимальное значение шкалы критериев. Нужно для того, 
            чтобы проверять наличие "соседа снизу" для данной вершины.
    """
    for u in G.nodes():
        for k in range(m):
            if u[k]>l:
                v=list(u)
                v[k]-=1
                G.add_edge(u,tuple(v))

def detect_cycles(n, rel):
    """Проверка на наличие ориентированных циклов, 
        где хотя бы у одной пары смежных вершин в цикле 
        нет второго ребра, направленного в другую сторону, 
        осуществляется при помощи алгоритма Беллмана-Форда.
    
        Мы просто ищем цикл отрицательного веса в графе и в
        случае успеха мы будем знать, что в введённых данных
        есть противоречие.
    """
    edges = []
    for i, j, r in rel:
        if r == 1:
            edges.append((i, j, -1))
        elif r == -1:
            edges.append((j, i, -1))
        else:
            edges.extend([(i, j, 0), (j, i, 0)])
            
    def bf(src):
        INF = sys.maxsize
        dist = [INF] * (n + 1)
        dist[src] = 0
        for _ in range(n - 1):
            updated = False
            for u, v, w in edges:
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True
            if not updated:
                break
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                return True
        return False
    
    return any(bf(i) for i in range(1, n + 1))

def input_qual(m):
    qual_rel=[]
    while True:
        k=n_input("Сколько пар качественных сравнений? ")
        print("""Введите через пробел пару критериев,
                а также отношение доминирования между ними 
                (1, если первый важнее второго
                -1, если второй важнее первого
                0, если они равноважны): """)
        print()
        for _ in range(k):
            i,j,r=map(int, input(f"Сообщение {_+1}: ").split())
            qual_rel.append((i,j,r))
        if not detect_cycles(m,qual_rel):
            break
        print("Противоречия найдены, введите снова.")
        qual_rel.clear()
    return qual_rel

def vec_space(dim, l, h):
    return [tuple(v) for v in product(range(l, h+1), repeat=dim)]

def add_qual_info(graph: Graph, i, j, rel):
    for node in graph.nodes():
        a, b = node[i-1], node[j-1]
        if a == b:
            continue
        node_2 = list(node)
        node_2[i-1], node_2[j-1] = b, a
        node_2 = tuple(node_2)
        if rel == 1:
            if a < b:
                graph.add_edge(node_2, node)
            else:
                graph.add_edge(node, node_2)
        elif rel == -1:
            if a < b:
                graph.add_edge(node, node_2)
            else:
                graph.add_edge(node_2, node)
        else:
            graph.add_edge(node, node_2)
            graph.add_edge(node_2, node)

def bfs_path(graph: Graph, start, end):
    if start == end:
        return [start]
    seen = set()
    seen.add(start)
    d = deque([(start, [start])])
    while d:
        cur, path = d.popleft()
        for w in graph.neighbors(cur):
            if w == end:
                return path + [w]
            if w not in seen:
                seen.add(w)
                d.append((w, path + [w]))
    return None

def filter_with_explanations(alts: list, graph):
    result = []
    for a in alts:
        dominated = False
        for b in alts:
            if a == b:
                continue
            path = bfs_path(graph, b, a)
            if path and not bfs_path(graph, a, b):
                print(f"Альтернатива [{alts.index(a)+1}] {a} доминируется через объясняющую цепочку: {path}")
                dominated = True
                break
        if not dominated:
            result.append(a)
    return result

#количественный этап 
def input_quant():
    m = n_input("Сколько пар количественных сравнений? ")
    print("""Введите через пробел пару критериев и степень доминирования 
          первого критерия над вторым (степень доминирования - положительное число): """)
    t = []
    for _ in range(m):
        i, j, h = input(f"Сообщение {_+1}: ").split()
        t.append((int(i), int(j), float(h)))
    return t

def check_consistency(n, qual, quant):
    rel = []

    for w in qual:
        rel.append(w)

    for i, j, h in quant:
        if h==1:
            rel.append((i,j,0))
        elif h>1:
            rel.append((i,j,1))
        else:
            rel.append((i,j,-1))

    return not detect_cycles(n, rel)

def lin_order(n, qual, quant):
    G = {i: set() for i in range(1, n+1)}
    for i,j,rel in qual:
        if rel==1:
            G[i].add(j)
        elif rel==-1:
            G[j].add(i)
        else:
            G[i].add(j); G[j].add(i)
    for i,j,h in quant:
        if h==1:
            G[i].add(j); G[j].add(i)
        elif h>1:
            G[i].add(j)
        else:
            G[j].add(i)
    def reachable(u, v):
        seen = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for y in G[x]:
                if y == v:
                    return True
                if y not in seen:
                    seen.add(y); stack.append(y)
        return False
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if not (reachable(i,j) or reachable(j,i)):
                return False
    return True

def build_beta_vector(n, quant):
    A=[]
    b=[]
    for i,j,h in quant:
        row=[0]*n
        row[i-1]=1
        row[j-1]=-1
        A.append(row)
        b.append(np.log(h))
    A=np.array(A)
    b=np.array(b)
    x,*_=np.linalg.lstsq(A,b,rcond=None)
    x=np.exp(x)
    return x/sum(x)

def compute_B_vectors(alts, beta, l, h):
    B_list = []
    for y in alts:
        Bk = {}
        for k in range(l, h+1):
            Bk[k] = sum(beta[i] for i, val in enumerate(y) if val >= k)
        B_list.append(Bk)
    return B_list

    
    

def main():
    print("=== Система поддержки принятия решений ===")
    m=n_input("Введите количество критериев: ")
    l, h=scale()
    print(f"Шкала: {l}–{h}\n")

    n= n_input("Сколько альтернатив? ")
    alts=[]
    
    cnt=n
    while cnt>0:
        vals = input(f"Альтернатива #{n-cnt+1} ({m} числа): ").split()
        if len(vals) != m:
            print("Неверное число значений.")
            continue
        t=tuple(map(int, vals))
        if t in alts:
            print("Это альтернатива уже предъявлена. Пожалуйста, введите новую альтернативу.")
            continue
        else:
            alts.append(tuple(map(int, vals)))
        cnt-=1
        
    print()
    print("Ваши альтернативы: ")
    for i in range(n):
        print(f" [{i+1}] {alts[i]}")

    print("\n--- Шаг 1: Парето-анализ ---")
    n_d_i=pareto(alts)
    print("Недоминируемые альтернативы:")
    for i in n_d_i:
        print(f" [{i+1}] {alts[i]}")
    if input("Результат устраивает? (y/n): ").lower()=="y":
        return
    
    non_dom=[alts[i] for i in n_d_i]




    print("\n--- Шаг 2: Качественная оценка ---")
    G = Graph()

    for vec in vec_space(m, l, h):
        G.add_node(vec)

    add_pareto_edges(G, m, l)

    qual_rel=input_qual(m)
    print()
    
    for i,j,r in qual_rel:
        add_qual_info(G,i,j,r)
    non_dom = filter_with_explanations(non_dom, G)
    print("\nНедоминируемые с учетом качественной информации:")
    for v in non_dom:
        print(f" [{alts.index(v)+1}] {v}")
    if input("Результат устраивает? (y/n): ").lower()=="y":
        return




    print("\n--- Шаг 3: Количественная оценка ---")
    while True:
        quant_rel = input_quant()
        if not check_consistency(m, qual_rel, quant_rel):
            print("Количественные данные противоречат качественным. Отредактируйте введённые данные и попробуйте снова.\n")
            continue
        if not lin_order(m, qual_rel, quant_rel):
            print("Недостаточно данных для полного ранжирования. Отредактируйте введённые данные и попробуйте снова.\n")
            continue
        break
    
    beta = build_beta_vector(m, quant_rel)
    print("\nВектор важностей β:")
    for idx, val in enumerate(beta, start=1):
        print(f" β[{idx}] = {val:.4f}")
    Bm = compute_B_vectors(alts, beta, l, h)
    
    hlp=[0]*n
    
    for i in range(len(non_dom)):
        hlp[alts.index(non_dom[i])]=1
    
    
    print("\n--- Сравнение альтернатив ---")
    for i in range(len(non_dom)):
        y = non_dom[i]
        for j in range(len(non_dom)):
            z = non_dom[j]
            if y!=z:
                Bi = Bm[alts.index(y)]
                Bj = Bm[alts.index(z)]
                better = all(Bi[k]>=Bj[k] for k in range(l, h+1))
                equal = all(Bi[k]==Bj[k] for k in range(l, h+1))
                if better and not equal:
                    print(f"{y} предпочтительнее {z}")
                    hlp[alts.index(z)]=0
                elif equal:
                    print(f"{y} эквивалентно {z}")
    
    non_dom=[alts[i] for i in range(n) if hlp[i]==1]
                    
    print("\nНедоминируемые с учетом количественной информации:")
    for v in non_dom:
        print(f" [{alts.index(v)+1}] {v}")
        
    print("\nАнализ завершён.")


if __name__ == '__main__':
    main()