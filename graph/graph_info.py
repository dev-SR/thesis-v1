from __future__ import annotations
import pandas as pd
from pandas import DataFrame
from collections import defaultdict, deque
from typing import List, DefaultDict, Deque, Dict
AdjacencyList = DefaultDict[str, List[str]]
Levels = List[List[Dict]]


class InfoManager:
    def __init__(self, file_path: str = None, df: DataFrame = None) -> None:
        if file_path:
            self.inf: DataFrame = pd.read_csv(file_path)
        elif df is not None:
            self.inf = df

    def get_info_by_uuid(self, id: str) -> List[dict]:
        f = self.inf[self.inf['uuid'] == id]
        return f.to_dict('records')[0]

    def get_info_by_uuidIds(self, ids: List[str]) -> List[dict]:
        f = self.inf[self.inf['uuid'].isin(ids)]
        return f.to_dict('records')

    def get_info_by_key(self, key: str, value: str) -> List[dict]:
        f = self.inf[self.inf[key] == value]
        return f.to_dict('records')

    def get_info_keys_by_uuid(self, id: str, keys: List[str]) -> Dict:
        f = self.inf[self.inf['uuid'] == id]
        return f.iloc[0][keys].to_dict()

    def get_refs_by_uuid(self, uuid: str) -> List[str]:
        f = self.inf[self.inf['parent_id'] == uuid]
        return f.to_dict('records')

    def get_parent_info_by_child_id(self, child_id: str) -> List[dict]:
        f = self.inf[self.inf['uuid'] == child_id]
        parent_id = f.iloc[0]['parent_id']
        p = self.inf[self.inf['uuid'] == parent_id]
        return p.to_dict('records')[0]


class GraphManager:
    def __init__(self) -> None:
        self.adjacencyListUnique: AdjacencyList = defaultdict(list)
        self.adjacencyListActual: AdjacencyList = defaultdict(list)
        self.levelOrderList: Levels = []
        self.levelOrderIdsList: Levels = []
        self.inf: DataFrame | None = None
        self.im: InfoManager | None = None

    @classmethod
    def init_from_csv(cls, csv_path: str, read_full: bool = False) -> GraphManager:
        g = cls()
        g.inf = pd.read_csv(csv_path)
        g.im = InfoManager(df=g.inf)

        print(g.inf.shape)
        for _, row in g.inf.iterrows():
            g.addEdge(row['parent_id'], row['uuid'], unique=True)
            g.addEdge(row['parent_paper_id'], row['paper_id'], unique=False)
        return g

    def addEdge(self, parent_id: str, child_id: str, unique) -> None:
        if unique:
            self.adjacencyListUnique[parent_id].append(child_id)
        else:
            self.adjacencyListActual[parent_id].append(child_id)

    def levelOrderFull(self, root: str) -> List[List[Dict]]:
        self.levelOrderList = []
        q: Deque = deque()
        q.append(root)
        while q:
            currentLevel: List[Dict] = []
            currentQLength = len(q)
            for _ in range(currentQLength):
                currentNode = q.popleft()
                # currentLevel.append(currentNode)
                p = self.im.get_info_keys_by_uuid(
                    currentNode, ['paper_id', 'uuid', 'parent_id'])
                currentLevel.append(p)
                for children in self.adjacencyListUnique[currentNode]:
                    q.append(children)

            self.levelOrderList.append(currentLevel)

        return self.levelOrderList

    def levelOrderIdsOnly(self, root: str) -> List[List[Dict]]:
        self.levelOrderIdsList = []
        q: Deque = deque()
        q.append(root)
        while q:
            currentLevel: List[Dict] = []
            currentQLength = len(q)
            for _ in range(currentQLength):
                currentNode = q.popleft()
                currentLevel.append(currentNode)
                for children in self.adjacencyListUnique[currentNode]:
                    q.append(children)

            self.levelOrderIdsList.append(currentLevel)

        return self.levelOrderIdsList

    def get_level_no_by_uid(self, levelsOrderList: List[List[Dict]], id: str) -> str:
        for i, l in enumerate(levelsOrderList):
            for d in l:
                if d['uuid'] == id:
                    return i
        return ""

    def get_levels_no_by_paper_id(self, levelsOrderList: List[List[Dict]], id: str) -> List[str]:
        inLevels = []
        for i, l in enumerate(levelsOrderList):
            for d in l:
                if d['paper_id'] == id:
                    inLevels.append(i)
        return inLevels

    # def alreadyProcessed(self, id: str, levels: List[List[str]]) -> bool:
    # 	print(len(levels))
    # 	for i in range(len(levels) - 1):
    # 		print(i)
    # 		# if id in levels[i]:
    # 		# return True

    def getPaths(self, s, d, unique=True) -> List[List[str]]:
        self.path = []
        self.ans = []

        def dfs(root):
            self.path.append(root)
            if root == d:
                if not unique:
                    self.ans.append(self.path)
                else:
                    self.ans = self.path

                return
            if not unique:
                for child in self.adjacencyListActual[root]:
                    dfs(child)
                    # after finishing traverse the child node, remove it from current path
                    self.path = self.path[:-1]
            else:
                for child in self.adjacencyListUnique[root]:
                    dfs(child)
                    # after finishing traverse the child node, remove it from current path
                    self.path = self.path[:-1]
        dfs(s)

        return self.ans
