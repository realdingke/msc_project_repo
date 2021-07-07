import datetime
from typing import Dict, List, Union

from tubemap.journey import Journey
from tubemap.tubemap import TubeMap


class User:
    MAX_PRICE_PER_DAY_WITH_PEAK = 8.80
    MAX_PRICE_PER_DAY_WITHOUT_PEAK = 7.00

    def __init__(self, name: str, tube_map: TubeMap):
        """

        self.journeys_per_date has the following form:
        {
            date_1: [
                journey_1_a,
                journey_1_b,
            ],
            date_2: [
                journey_2_a,
            ],
        }
        where date_1 and date_2 are of type datetime.date,
        and journey_1_a, journey_1_b and journey_2_a are of type Journey.

        :param name: full name of the user
        :param tube_map: entire tube map that is used
        """

        self.name = name
        self.tube_map = tube_map

        self.journeys_per_date = dict()  # type: Dict[datetime.date, List[Journey]]

    def register_journey(self,
                         time_start: datetime.time,
                         time_end: datetime.time,
                         date: datetime.date,
                         list_successive_stations: List[str],
                         ) -> Union[Journey, None]:
        """
        Register a journey in self.journeys_per_date
        :param time_start: the time at which the journey started
        :param time_end: the time at which the journey ended
        :param date: the date when that journey was performed
        :param list_successive_stations: list of the successive stations
        :return: - if no JourneyNotValid exception is raised when creating the Journey, then the newly registered
        journey is added in self.journeys_per_date.
                 - if a JourneyNotValid is raised during the creation of the Journey, then return None.
        """

        # TODO
        try:
            journey = Journey(time_start=time_start,
                      time_end=time_end,
                      date=date,
                      list_successive_stations=list_successive_stations,
                      tube_map=self.tube_map)
        except:
            return None
        
        try:
            length = len(self.journeys_per_date[date])
            self.journeys_per_date[date].append(journey)
            return journey
        except KeyError:
            self.journeys_per_date[date] = []
            self.journeys_per_date[date].append(journey)
            return journey


    def compute_price_per_day(self, date: datetime.date) -> float:
        """
        :param date: day at which we want to calculate the price
        :return: Total amount of money spent in the tube by the user at that day,
        if the user did not use the tube on that date, then the function should return 0.
        """

        # TODO
        try:
            length = len(self.journeys_per_date[date])
        except KeyError:
            return 0
        
        journeys = self.journeys_per_date[date]
        peak_list = []  #list of bools
        total_price = 0
        for journey in journeys:
            total_price += journey.compute_price()
            peak_list.append(journey.is_on_peak())
        
        #determine the price cap and get the actual price
        if True in peak_list:
            threshold = User.MAX_PRICE_PER_DAY_WITH_PEAK
            if total_price < threshold:
                return total_price
            else:
                return threshold
        else:
            threshold = User.MAX_PRICE_PER_DAY_WITHOUT_PEAK
            if total_price < threshold:
                return total_price
            else:
                return threshold


if __name__ == '__main__':
    tube_map = TubeMap()
    tube_map.import_tube_map_from_json("data/london.json")

    user = User("Bob", tube_map)

    # A journey on the 30/10/2019 off peak
    user.register_journey(time_start=datetime.time(hour=12, minute=15),
                          time_end=datetime.time(hour=12, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )

    # Another journey on the 30/10/2019 on peak
    user.register_journey(time_start=datetime.time(hour=18, minute=15),
                          time_end=datetime.time(hour=18, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )
    
    print(user.compute_price_per_day(date=datetime.date(year=2019, month=10, day=30)))

    # Trying to add an Invalid journey (the function should return None in that case)
    print(user.register_journey(time_start=datetime.time(hour=18, minute=15),
                          time_end=datetime.time(hour=18, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'South Kensington'], ))

    print(user.compute_price_per_day(date=datetime.date(year=2019, month=10, day=30)))

    # Adding more journeys to reach the maximum price per day
    user.register_journey(time_start=datetime.time(hour=8, minute=15),
                          time_end=datetime.time(hour=8, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )

    user.register_journey(time_start=datetime.time(hour=10, minute=15),
                          time_end=datetime.time(hour=10, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )

    print(user.compute_price_per_day(date=datetime.date(year=2019, month=10, day=30)))
