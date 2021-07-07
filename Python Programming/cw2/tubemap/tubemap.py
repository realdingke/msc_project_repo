from typing import Set, List
import json
import os
import random


class StationDoesNotExist(Exception):
    def __init__(self, name_element):
        self.message = f"The station '{name_element}' does not exist in the Loaded tube map."
        super(StationDoesNotExist, self).__init__(self.message)


class TubeMap(object):
    """
    This class has two main attributes:
    - graph_tube_map
    - set_zones_per_station

    The attribute graph_tube_map should have the following form:

    {
        "station_A": {
            "neighbour_station_1": [
                {
                    "line": "name of a line between station_A and neighbour_station_1",
                    "time": "time it takes in minutes to go from station_A to neighbour_station_1 WITH THAT line"
                },
                {
                    "line": "name of ANOTHER line between station_A and neighbour_station_1",
                    "time": "time it takes in minutes to go from station_A to neighbour_station_1 with that OTHER line"
                }
            ],
            "neighbour_station_2": [
                {
                    "line": "name of line between station_A and neighbour_station_2",
                    "time": "time it takes in minutes to go from station_A to neighbour_station_2"
                }
            ]
        }

        "station_B": {
            ...
        }

        ...

    }

        Also, for instance:
        self.graph_tube_map['Hammersmith'] should be equal to:
        {
            'Barons Court': [
                {'line': 'District Line', 'time': 1},
                {'line': 'Piccadilly Line', 'time': 2}
            ],
            'Ravenscourt Park': [
                {'line': 'District Line', 'time': 2}
            ],
            'Goldhawk Road': [
                {'line': 'Hammersmith & City Line', 'time': 2}
            ],
            'Turnham Green': [
                {'line': 'Piccadilly Line', 'time': 2}
            ]
        }

    The attribute set_zones_per_station should have the following form:
    {
        station_1: {zone_a},
        station_2: {zone_a, zone_b},
        ...
    }
    For example, with the London tube map,
    self.set_zones_per_station["Turnham Green"] == {2, 3}
    """

    TIME = "time"
    LINE = "line"
    ZONES = "zones"
    CONNECTIONS = "connections"

    def __init__(self):
        self.graph_tube_map = dict()
        self.set_zones_per_station = dict()

    def import_tube_map_from_json(self, file_path: str) -> None:
        """
        Import the tube map information from a JSON file.
        During that import, the two following attributes should be updated:
        - graph_tube_map
        - set_zones_per_station

        :param file_path: relative or absolute path to the json file containing all the information about the
        tube map graph to import
        """
        
        # TODO
        path = os.path.abspath(file_path)
        with open(path) as f:
            data = json.load(f)

        lines = data['lines']
        stations = data['stations']
        connections = data['connections']
        
        #index to name translation dict for stations
        station_name_dict = {}
        for dct in stations:
            station_id = dct['id']
            station_name = dct['name']
            station_name_dict[station_id] = station_name

        #index to name translation dict for lines
        line_name_dict = {}
        for dct in lines:
            line_id = dct['line']
            line_name = dct['name']
            line_name_dict[line_id] = line_name

        #initialize the tube_map for the first attribute
        tube_map = {}
        for name in station_name_dict.values():
            tube_map[name] = {}
            
        #connections
        for main_id in station_name_dict.keys():
            neighbor_stats_list = []
            previous_neighbors = []
            for dct in connections:
                if main_id == dct['station1']:
                    neighbor_id = dct['station2']
                    dct_copy = dct.copy()
                    dct_copy.pop('station1')
                    dct_copy.pop('station2')
                    dct_copy['line'] = line_name_dict[dct_copy['line']]
                    dct_copy['time'] = int(dct_copy['time'])
                    if neighbor_id not in previous_neighbors:
                        tube_map[station_name_dict[main_id]][station_name_dict[neighbor_id]] = []
                    if dct_copy not in tube_map[station_name_dict[main_id]][station_name_dict[neighbor_id]]:
                        tube_map[station_name_dict[main_id]][station_name_dict[neighbor_id]].append(dct_copy)
                    previous_neighbors.append(neighbor_id)
                elif main_id == dct['station2']:
                    neighbor_id = dct['station1']
                    dct_copy = dct.copy()
                    dct_copy.pop('station1')
                    dct_copy.pop('station2')
                    dct_copy['line'] = line_name_dict[dct_copy['line']]
                    dct_copy['time'] = int(dct_copy['time'])
                    if neighbor_id not in previous_neighbors:
                        tube_map[station_name_dict[main_id]][station_name_dict[neighbor_id]] = []
                    if dct_copy not in tube_map[station_name_dict[main_id]][station_name_dict[neighbor_id]]:
                        tube_map[station_name_dict[main_id]][station_name_dict[neighbor_id]].append(dct_copy)
                    previous_neighbors.append(neighbor_id)

        #initialize the station_zones for the second attribute
        station_zones = {}
        for name in station_name_dict.values():
            station_zones[name] = set()

        #stations
        for dct in stations:
            zone_str = dct['zone']
            try:
                zone_num = int(zone_str)
            except:
                zone_num = float(zone_str)
            
            if type(zone_num) != float:
                station_zones[dct['name']].add(zone_num)
            else:
                first_zone = int(round(zone_num))
                if first_zone > zone_num:
                    second_zone = first_zone - 1
                else:
                    second_zone = first_zone + 1
                station_zones[dct['name']].update({first_zone, second_zone})
        
        #update the attributes
        self.graph_tube_map = tube_map
        self.set_zones_per_station = station_zones
        pass
    
    def argmin(self, vertices, data):
        """argmin with random tie-breaking for choosing unvisited vertex in Dijkstra algorithm
        """
        lowest = float("inf")
        ties = []

        for i in vertices:
            if data[i]['dist'] < lowest:
                lowest = data[i]['dist']
                ties = []

            if data[i]['dist'] == lowest:
                ties.append(i)

        return random.choice(ties)


    def get_fastest_path_between(self, station_start: str, station_end: str) -> List[str]:
        """
        Implementation of Dijkstra algorithm to find the fastest path from station_start to station_end

        for instance: get_fastest_path_between('Stockwell', 'South Kensington') should return the list:
        ['Stockwell', 'Vauxhall', 'Pimlico', 'Victoria', 'Sloane Square', 'South Kensington']

        See here for more information: https://en.wikipedia.org/wiki/Dijkstra's_algorithm#Pseudocode

        :param station_start: name of the station at the beginning of the journey
        :param station_end: name of the station at the end of the journey
        :return: An ordered list representing the successive stations in the fastest path.
        :raise StationDoesNotExist if the station is not in the loaded tube map
        """
        
        # TODO
        if station_start not in self.graph_tube_map.keys():
            raise StationDoesNotExist(station_start)
        elif station_end not in self.graph_tube_map.keys():
            raise StationDoesNotExist(station_end)
        
        #initialize unvisited vertex set Q, vertex attributes dict
        vertex_set = set()
        vertex_data = {}
        for station in self.graph_tube_map.keys():
            vertex_set.add(station)
            vertex_data[station] = {}
            vertex_data[station]['dist'] = float('inf')
            vertex_data[station]['prev'] = None
        
        #set source vertex(starting position) to have 0 dist
        vertex_data[station_start]['dist'] = 0
        
        #start the Dijkstra algorithm
        while len(vertex_set) > 0:
            vertex_to_visit = self.argmin(vertex_set, vertex_data)
            vertex_set.discard(vertex_to_visit)
            if vertex_to_visit == station_end:
                break
            
            #find all neighbors of the vertex to visit
            for neighbor, line_list in self.graph_tube_map[vertex_to_visit].items():
                alt_minimum = float("inf")
                for dct in line_list:
                    alt = vertex_data[vertex_to_visit]['dist'] + int(dct['time'])
                    if alt < alt_minimum:
                        alt_minimum = alt
                if alt_minimum < vertex_data[neighbor]['dist']:
                    vertex_data[neighbor]['dist'] = alt_minimum
                    vertex_data[neighbor]['prev'] = vertex_to_visit
                
        #generate a trace(list of stations) of shortest path to target
        shortest_path = []
        u = station_end
        if (vertex_data[u]['prev'] is not None) or (u == station_start):
            while u is not None:
                shortest_path.append(u)
                u = vertex_data[u]['prev']
        
        return shortest_path[::-1]


    def get_set_lines_for_station(self, name_station: str) -> Set[str]:
        """
        :param name_station: name of a station in the tube map. (e.g. 'Hammersmith')
        :return: set of the names of the lines on which the station is found.
        :raise StationDoesNotExist if the station is not in the loaded tube map
        """

        # TODO
        line_names = set()
        if name_station in self.graph_tube_map.keys():
            for station in self.graph_tube_map[name_station].keys():
                list_of_lines = self.graph_tube_map[name_station][station]
                for line in list_of_lines:
                    line_names.add(line['line'])
        else:
            raise StationDoesNotExist(name_station)
        
        return line_names


    def get_set_all_stations_on_line(self, line: str) -> Set[str]:
        """
        :param line: name of a metro line (e.g. 'Victoria Line')
        :return: the set of all the stations on that line.
        """

        # TODO
        station_names = set()
        for main_station, neighbor_dict in self.graph_tube_map.items():
            for neighbor_station, line_lists in neighbor_dict.items():
                for dct in line_lists:
                    if line == dct['line']:
                        station_names.update(set([main_station, neighbor_station]))
            
        return station_names


if __name__ == '__main__':
    tube_map = TubeMap()
    tube_map.import_tube_map_from_json("data/london.json")
    print(tube_map.graph_tube_map["Hammersmith"],'\n')
    #print(tube_map.get_set_lines_for_station("Hammersmith"),'\n')
    #print(tube_map.get_set_all_stations_on_line("Piccadilly Line"))
    print(tube_map.get_fastest_path_between("Warren Street", "South Kensington"))
    #print(tube_map.set_zones_per_station["Turnham Green"])
