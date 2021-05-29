#import pygame
from gym_map.envs.map_constants import (DEFAULT_MAP_DATA, MAX_WIDTH,
    MAX_HEIGHT, MAP_BOUNDS, MAX_CHECKPOINTS, MAX_TELEPORTS, TILE_DICT, 
    MAP_TYPE_DICT, MAX_WALLS, MAX_SCORE, Tile)
import numpy as np
import json
#from queue import PriorityQueue
from heapq import heappush, heappop

class Map:
    def __init__(self, map_file=None, render=True, calc_score_on_step=True):
        
        if map_file:
            with open(map_file, 'r') as fh:
                self.map_data = json.loads(fh.read())
        else:
            self.map_data = json.loads(DEFAULT_MAP_DATA)
        self.calc_score_on_step = calc_score_on_step
        self.width = self.map_data['width']
        self.height = self.map_data['height']
        self.walls_remaining = self.map_data['walls']
        self.moves = set()
        self.path1 = []
        self.path2 = []
        
        self._load_map(self.width, self.height, self.map_data['tiles'])
        self._create_observation_space()
        self._load_fixed_data()
        self.reset()
        self.map_form = 0
        self.score_history = {}
   
    def _load_map(self, data_width, data_height, tiles):
        self.map_cells = np.zeros(MAP_BOUNDS, dtype=int)
        
        # TODO - INPUT LAYER IS ONLY THE SMALL MAP
        #self.input_layer = np.zeros((Tile.NUM_TILES, self.width, self.height), dtype=np.int)
        self.input_layer = np.zeros((Tile.NUM_TILES, MAX_WIDTH, MAX_HEIGHT), dtype=np.int)
        letter = None
        num = None
        for y in range(MAX_HEIGHT):
            for x in range(MAX_WIDTH):
                if x >= data_width or y >= data_height:
                    letter = 'r'
                    num = 3
                else:
                    letter, num = tiles[y][x]
                t = TILE_DICT[letter][num] 
                self.map_cells[y,x] = t
                if x < data_width and y < data_height:
                    self.input_layer[t,x,y] = 1
           
    def _create_observation_space(self):
        self.state = {
            # Fixed
            'map_type': MAP_TYPE_DICT[self.map_data['name']],
            'num_checkpoints': self.map_data['checkpoints'],
            'num_teleports': self.map_data['teleports'],
            'num_walls': int(self.map_data['walls']),
            # Dynamic
            'teleports_used': 0,
            'walls_remaining': int(self.map_data['walls']),
            'score': 0,
            'map': self.map_cells
        }
    
    def step(self, action=None, x=None, y=None):
        if action is not None:
            y = action[1]
            x = action[0]
            #print("(%s,%s)" % (x,y))
        move = x + y * self.width
        self.map_form ^= 2**move
        recalculate = False
        if self.map_cells[y,x] in (Tile.EMPTY, Tile.ROCK_2):
            if move in self.moves:
                self.moves.remove(move)
                self.state['walls_remaining'] += 1
                self.map_cells[y,x] = Tile.EMPTY
                self.input_layer[Tile.ROCK_2, x, y] = 0
                self.input_layer[Tile.EMPTY, x, y] = 1
                recalculate = True
            elif self.state['walls_remaining'] > 0:
                self.moves.add(move)
                self.state['walls_remaining'] -= 1
                self.map_cells[y,x] = Tile.ROCK_2
                self.input_layer[Tile.EMPTY, x, y] = 0
                self.input_layer[Tile.ROCK_2, x, y] = 1
                recalculate = (x, y) in self.on_the_path
            if self.calc_score_on_step and recalculate:
                self.calc_score()
            else:
                if self.state['walls_remaining'] == 0:
                    if self.map_form not in self.score_history:
                        self.calc_score()
                        self.score_history[self.map_form] = self.state['score']
                    else:
                        self.state['score'] = self.score_history[self.map_form]
    
    def reset(self):
        self.state['teleports_used'] = 0
        self.state['walls_remaining'] = int(self.map_data['walls'])
        for move in self.moves:
            y = int(move / self.width)
            x = int(move % self.width)
            self.map_cells[y,x] = Tile.EMPTY
            self.input_layer[Tile.ROCK_2, x, y] = 0
            self.input_layer[Tile.EMPTY, x, y] = 1
        self.moves = set()
        self.calc_score()
        self.map_form = 0
        return self.state

    def calc_score(self):
        self.path1, self.path2, num_teleports = self._calc_path()
        self.state['score'] = len(self.path1) + len(self.path2)
        self.on_the_path = set(self.path1) | set(self.path2)
        self.path_list = list(self.on_the_path)
        self.state['teleports_used'] = num_teleports


    def _load_fixed_data(self):
        self.fixed_map_data = {
            'start': [],
            'finish': [],
            'targets': [[] for _ in range(self.map_data['checkpoints'])]
        }
        for t in range(self.map_data['teleports']):
            self.fixed_map_data[t+Tile.OUT_1] = None
        for y in range(self.height):
            for x in range(self.width):
                t = self.map_cells[y,x]
                if t >= Tile.OUT_1 and t <= Tile.OUT_7:
                    self.fixed_map_data[t] = (x, y)
                if t in (Tile.G_START, Tile.R_START):
                    self.fixed_map_data['start'].append((x, y, t))
                if t == Tile.FINISH:
                    self.fixed_map_data['finish'].append((x, y))
                if t >= Tile.A and t <= Tile.N:
                    self.fixed_map_data['targets'][t-Tile.A].append((x, y))
        return self.fixed_map_data
        
    def _calc_path(self):
        path1, teleports1 = self._calc_path_helper(Tile.G_START)
        path2, teleports2 = self._calc_path_helper(Tile.R_START)
        if path1 is None or path2 is None:
            return [], [], 0
        else:
            return path1, path2, (teleports1+teleports2)
    
    def calc_path_helper(self, start):
        pass
    
    ##### REDOING EVERYTHING BELOW

    def _calc_path_helper(self, start):
        map_cells = self.map_cells
        width = self.width
        height = self.height
        data = self.fixed_map_data
        
        targets = None
        if start == Tile.R_START:
            targets = data['targets'][::-1] + [data['finish']]
        else:
            targets = data['targets'] + [data['finish']]
        #print(targets)
        starts = [(s[0], s[1]) for s in data['start'] if s[2] == start]
        last_x = 0
        last_y = 0
        full_path = []
        state = 0
        target_index = 0
        max_targets = len(targets)
        ins = set(Tile.INS)
        if starts:
            while target_index < max_targets:
                best_path_len = MAX_SCORE
                best_path = None
                best_x = None
                best_y = None
                checkpoints = targets[target_index]
                for x, y in checkpoints:
                    distances = np.full((self.height, self.width), MAX_SCORE, dtype=int)
                    distances[y,x] = 0
                    self._d(distances, x, y, start, last_x, last_y, state != 0)
                    if state == 0:
                        path = self._get_path(distances, starts=starts)
                    else:
                        path = self._get_path(distances, x=last_x, y=last_y)
                    if path is not None:
                        if best_path_len == MAX_SCORE or best_path_len > len(path):
                            best_path = path
                            best_path_len = len(path)
                            best_x = x
                            best_y = y
                            #print(distances)
                            #print(distances[5:8, 12:16])
                if best_path is None:
                    return None, 0
                i = 0
                tp = False
                for x, y in best_path:
                    t = self.map_cells[y,x]
                    if t in ins:
                        #print('ah tp %s' % (t - Tile.IN_1 + 1))
                        ins.remove(t)
                        best_x, best_y = self.fixed_map_data[t-Tile.IN_1+Tile.OUT_1]
                        best_path = best_path[:i+1]
                        tp = True
                        break
                    i += 1
                last_x = best_x
                last_y = best_y
                state += 1
            
                #test = np.full((self.height, self.width), 8)
                #c = 0
                #for x, y in best_path:
                #    test[y][x] = c
                #    c += 1
                #print(test)
                if not tp:
                    target_index += 1
                #print(best_path[-8:])
                #print(len(best_path[1:]))
                full_path.extend(best_path[1:])
                #score += len(best_path) - 1
        #print(score)
        return full_path, (len(Tile.INS) - len(ins))
        
    def _d(self, distances, x, y, start, target_x, target_y, is_single):
        #q = PriorityQueue()
        #q.put((0, x, y))
        q = []
        heappush(q, (0, x, y))
        visited = np.full((self.height, self.width), 0, dtype=int)
        while q:#not q.empty():
            #c, x, y = q.get()
            c, x, y = heappop(q)
            visited[y,x] = 1
            if is_single and x == target_x and y == target_y:
                break
            if (self.map_cells[y,x] in Tile.ROCKS
                or (start == Tile.G_START and self.map_cells[y,x] == Tile.R_ALLOW)
                or (start == Tile.R_START and self.map_cells[y,x] == Tile.G_ALLOW)):
                distances[y,x] = MAX_SCORE
            else:
                self._add_adj(q, distances, x, y, visited, start)
        #print(distances[1:4,6:9])
        #print()
        #print(distances)

    def _add_adj(self, q, distances, x, y, visited, start):
        cost = distances[y,x]
        self._safe_add(q, distances, x, y-1, x, y-2, cost, visited, start)
        self._safe_add(q, distances, x+1, y, x+2, y, cost, visited, start)
        self._safe_add(q, distances, x, y+1, x, y+2, cost, visited, start)
        self._safe_add(q, distances, x-1, y, x-2, y, cost, visited, start)

    def _safe_add(self, q, distances, x, y, next_x, next_y, cost, visited, start):
        if x >= 0 and y >= 0 and x < self.width and y < self.height:
            if visited[y,x] != 0:
                return
            current_cost = distances[y,x]
            if self.map_cells[y,x] == Tile.ICE:
                delta_x = next_x - x
                delta_y = next_y - y
                x1 = next_x
                y1 = next_y
                cost1 = cost + 1
                while (x1 >= 0 and y1 >= 0 and x1 < self.width and y1 < self.height):
                    if self.map_cells[y1,x1] == Tile.ICE:
                        x1 += delta_x
                        y1 += delta_y
                        cost1 += 1
                    elif (self.map_cells[y1,x1] in Tile.ROCKS
                            or (start == Tile.G_START and self.map_cells[y1,x1] == Tile.R_ALLOW)
                            or (start == Tile.R_START and self.map_cells[y1,x1] == Tile.G_ALLOW)):
                        break
                    elif (distances[y1,x1] == MAX_SCORE or distances[y1,x1] > (cost1 + 1)):
                        #q.put((cost1+1, x1, y1
                        heappush(q, (cost1+1, x1, y1))
                        distances[y1,x1] = cost1 + 1
                        break
                    else:
                        break
            else:
                if (current_cost == MAX_SCORE or current_cost > (cost + 1)):
                    #q.put((cost+1, x, y))
                    heappush(q, (cost+1, x, y))
                    distances[y,x] = cost + 1

    def _get_path(self, distances, starts=None, x=None, y=None):
        if x == None:
            max_val = MAX_SCORE
            x = None
            y = None
            for start_x, start_y in starts:
                if max_val == MAX_SCORE or max_val > distances[start_y,start_x]:
                    x = int(start_x)
                    y = int(start_y)
                    max_val = distances[y,x]
        if distances[y,x] == MAX_SCORE:
            return None
        d = int(distances[y,x])
        path = [(x, y)]
        for dist in range(d-1, -1, -1):
            n = self._safe_check(distances, x, y-1, x, y-2, dist)
            if n > 0:
                for m in range(n):
                    path.append((x, y-1))
                    y = y - 1
            else:
                n = self._safe_check(distances, x+1, y, x+2, y, dist)
                if n > 0:
                    for m in range(n):
                        path.append((x+1, y))
                        x = x + 1
                else:
                    n = self._safe_check(distances, x, y+1, x, y+2, dist)
                    if n > 0:
                        for m in range(n):
                            path.append((x, y+1))
                            y = y + 1
                    else:
                        n = self._safe_check(distances, x-1, y, x-2, y, dist)
                        if n > 0:
                            for m in range(n):
                                path.append((x-1, y))
                                x = x - 1
        return path
        
    def _safe_check(self, distances, x, y, next_x, next_y, dist):
        if (x >= 0 and y >= 0 and x < self.width and y < self.height 
                    and (distances[y,x] == dist or self.map_cells[y,x] == Tile.ICE)):
            if self.map_cells[y,x] == Tile.ICE:
                delta_x = next_x - x
                delta_y = next_y - y
                safe = self._safe_check(distances, next_x, next_y, next_x + delta_x, next_y + delta_y, dist - 1)
                if safe > 0:
                    return safe + 1
                else:
                    return 0
            else:
                return 1
        else:
            return 0