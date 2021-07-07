from typing import List, Tuple

from battleship.ship import Ship

import random

OFFSET_UPPER_CASE_CHAR_CONVERSION = 64

class Board(object):
    """
    Class representing the board of the player. Interface between the player and its ships.
    """
    SIZE_X = 10  # length of the rectangular board, along the x axis
    SIZE_Y = 10  # length of the rectangular board, along the y axis

    # dict: length -> number of ships of that length
    DICT_NUMBER_SHIPS_PER_LENGTH = {1: 1,
                                    2: 1,
                                    3: 1,
                                    4: 1,
                                    5: 1}

    def __init__(self,
                 list_ships: List[Ship]):
        """
        :param list_ships: list of ships for the board.
        :raise ValueError if the list of ships is in contradiction with Board.DICT_NUMBER_SHIPS_PER_LENGTH.
        :raise ValueError if there are some ships that are too close from each other
        """

        self.list_ships = list_ships
        self.set_coordinates_previous_shots = set()

        if not self.lengths_of_ships_correct():
            total_number_of_ships = sum(self.DICT_NUMBER_SHIPS_PER_LENGTH.values())

            error_message = f"There should be {total_number_of_ships} ships in total:\n"

            for length_ship, number_ships in self.DICT_NUMBER_SHIPS_PER_LENGTH.items():
                error_message += f" - {number_ships} of length {length_ship}\n"

            raise ValueError(error_message)

        if self.are_some_ships_too_close_from_each_other():
            raise ValueError("There are some ships that are too close from each other.")

    def has_no_ships_left(self) -> bool:
        """
        :return: True if and only if all the ships on the board have sunk.
        """
        # TODO
        has_sunk_list = [ship.has_sunk() for ship in self.list_ships]
        if (False not in has_sunk_list):
            return True
        else:
            return False

    def is_attacked_at(self, coord_x: int, coord_y: int) -> Tuple[bool, bool]:
        """
        The board receives an attack at the position (coord_x, coord_y).
        - if there is no ship at that position -> nothing happens
        - if there is a ship at that position -> it is damaged at that coordinate

        :param coord_x: integer representing the projection of a coordinate on the x-axis
        :param coord_y: integer representing the projection of a coordinate on the y-axis
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where an
                    opponent's ship is.
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """

        # TODO
        self.set_coordinates_previous_shots.add((coord_x, coord_y))
        is_ship_hit = False
        has_ship_sunk = False
        for ship in self.list_ships:
            if (coord_x, coord_y) in ship.get_all_coordinates():
                is_ship_hit = True
                ship.gets_damage_at(coord_x, coord_y)
                if ship.has_sunk():
                    has_ship_sunk = True
        return (is_ship_hit, has_ship_sunk)

    def print_board_with_ships_positions(self) -> None:
        array_board = [[' ' for _ in range(self.SIZE_X)] for _ in range(self.SIZE_Y)]

        for x_shot, y_shot in self.set_coordinates_previous_shots:
            array_board[y_shot - 1][x_shot - 1] = 'O'

        for ship in self.list_ships:
            if ship.has_sunk():
                for x_ship, y_ship in ship.set_all_coordinates:
                    array_board[y_ship - 1][x_ship - 1] = '$'
                continue

            for x_ship, y_ship in ship.set_all_coordinates:
                array_board[y_ship - 1][x_ship - 1] = 'S'

            for x_ship, y_ship in ship.set_coordinates_damages:
                array_board[y_ship - 1][x_ship - 1] = 'X'

        board_str = self._get_board_string_from_array_chars(array_board)

        print(board_str)

    def print_board_without_ships_positions(self) -> None:
        array_board = [[' ' for _ in range(self.SIZE_X)] for _ in range(self.SIZE_Y)]

        for x_shot, y_shot in self.set_coordinates_previous_shots:
            array_board[y_shot - 1][x_shot - 1] = 'O'

        for ship in self.list_ships:
            if ship.has_sunk():
                for x_ship, y_ship in ship.set_all_coordinates:
                    array_board[y_ship - 1][x_ship - 1] = '$'
                continue

            for x_ship, y_ship in ship.set_coordinates_damages:
                array_board[y_ship - 1][x_ship - 1] = 'X'

        board_str = self._get_board_string_from_array_chars(array_board)

        print(board_str)

    def _get_board_string_from_array_chars(self, array_board: List[List[str]]) -> str:
        list_lines = []

        array_first_line = [chr(code + OFFSET_UPPER_CASE_CHAR_CONVERSION) for code in range(1, self.SIZE_X + 1)]
        first_line = ' ' * 6 + (' ' * 5).join(array_first_line) + ' \n'

        for index_line, array_line in enumerate(array_board, 1):
            number_spaces_before_line = 2 - len(str(index_line))
            space_before_line = number_spaces_before_line * ' '
            list_lines.append(f'{space_before_line}{index_line} |  ' + '  |  '.join(array_line) + '  |\n')

        line_dashes = '   ' + '-' * 6 * self.SIZE_X + '-\n'

        board_str = first_line + line_dashes + line_dashes.join(list_lines) + line_dashes

        return board_str

    def lengths_of_ships_correct(self) -> bool:
        """
        :return: True if and only if there is the right number of ships of each length, according to
        Board.DICT_NUMBER_SHIPS_PER_LENGTH
        """
        # TODO
        ship_dict = {}
        for ship in self.list_ships:
            ship_dict[ship.length()] = 1
        if ship_dict == Board.DICT_NUMBER_SHIPS_PER_LENGTH:
            return True
        else:
            return False

    def are_some_ships_too_close_from_each_other(self) -> bool:
        """
        :return: True if and only if there are at least 2 ships on the board that are near each other.
        """
        # TODO
        ship_close_list = []
        list_ships = self.list_ships.copy()     #prevent altering the attributes as list is mutable
        while len(list_ships) > 1:
            for idx, ship in enumerate(list_ships):
                if idx > 0:
                    ship_close_list.append(ship.is_near_ship(list_ships[0]))
            list_ships.remove(list_ships[0])
        if (True in ship_close_list):
            return True
        else:
            return False
        


class BoardAutomatic(Board):
    def __init__(self):
        self.list_check = False
        super().__init__(list_ships=self.generate_ships_automatically())
    
    
    def vertical_or_horizontal(self):
        """
        return: a randomly chosen boolean, True means vertical, False is horizontal
        """
        #
        return random.choice([True, False])
    
    def is_out_of_bound(self, coordinates):
        """
        return: a boolean that is True if and only if the input coordinates are not within the boundaries
        """
        #
        if (coordinates[0] in list(range(1, Board.SIZE_X+1))) and (coordinates[1] in list(range(1, Board.SIZE_Y+1))):
            return False
        else:
            return True
    
    def generate_test_ships_automatically(self):
        """
        every time upon calling, populates a test list of automatically (randomly) generated ships for the board
        return: a test list of ships
        """
        #
        test_list_ships = []
        for length, num in Board.DICT_NUMBER_SHIPS_PER_LENGTH.items():
            if self.vertical_or_horizontal():    
                #make a vertical ship
                bound_detect = True
                while bound_detect:
                    start = (random.randint(1, Board.SIZE_X), random.randint(1, Board.SIZE_Y))
                    end = (start[0], start[1]+length-1)
                    bound_detect = self.is_out_of_bound(end)
                test_list_ships.append(Ship(start, end))
            else:
                #make a horizontal ship
                bound_detect = True
                while bound_detect:
                    start = (random.randint(1, Board.SIZE_X), random.randint(1, Board.SIZE_Y))
                    end = (start[0]+length-1, start[1])
                    bound_detect = self.is_out_of_bound(end)
                test_list_ships.append(Ship(start, end))
        return test_list_ships

    def check_list_ships(self, list_ships):
        """to check whether if some in test list of ships are too close to each other
        """
        #
        try:
            test_board = Board(list_ships)
        except:
            self.list_check = False
        else:
            self.list_ships = list_ships
            self.list_check = True

    def generate_ships_automatically(self) -> List[Ship]:
        """
        :return: A list of automatically (randomly) generated ships for the board
        """
        # TODO
        while not self.list_check:
            self.check_list_ships(self.generate_test_ships_automatically())
        return self.list_ships
            

if __name__ == '__main__':
    # SANDBOX for you to play and test your functions
    #list_ships = [Ship(coord_start=(3,5), coord_end=(3,5)), Ship(coord_start=(6,8), coord_end=(6,9)), Ship(coord_start=(4,5), coord_end=(6,5)), Ship(coord_start=(6,8), coord_end=(9,8)), Ship(coord_start=(5,2), coord_end=(5,6))]

    #board = Board(list_ships)
    #board.print_board_with_ships_positions()
    #board.print_board_without_ships_positions()
    #print(board.is_attacked_at(5, 4),board.is_attacked_at(1, 1))
    #print(board.set_coordinates_previous_shots)
    #print(board.has_no_ships_left())
    #print(board.lengths_of_ships_correct())
    #print(board.are_some_ships_too_close_from_each_other())
    #print(board.list_ships)
    b = BoardAutomatic()
    #print(b.list_ships)
    print(b.lengths_of_ships_correct())
    b.print_board_with_ships_positions()
