from functools import total_ordering
from assignment1 import FloorGraph, restaurantFinder
import unittest

class TestRestaurantFinder(unittest.TestCase):

    def test_1(self):
        d = 0
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 434) # Expected total annual revenue
        self.assertEqual(selected_sites, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Expected selected site numbers

    def test_2(self):
        d = 1
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 252) # Expected total annual revenue
        self.assertEqual(selected_sites, [1,4,6,8,10]) # Expected selected site numbers

    def test_3(self):
        d = 2
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 245) # Expected total annual revenue
        self.assertEqual(selected_sites, [1,4,7,10]) # Expected selected site numbers

    def test_4(self):
        d = 3
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 175) # Expected total annual revenue
        self.assertEqual(selected_sites, [1,6,10]) # Expected selected site number:

    def test_5(self):
        d = 7
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 100) # Expected total annual revenue
        self.assertEqual(selected_sites, [7]) # Expected selected site numbersdef test_single_site(self):


    def test_6(self):
        d = 1
        site_list = [50, -10, 12, -65, -40, 95, -100, 12, -20, -30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 169) # Expected total annual revenue
        self.assertEqual(selected_sites, [1,3,6,8]) # Expected selected site numbers 


    def test_7(self): 
        d = 2
        site_list = [5]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 5) # Expected total annual revenue
        self.assertEqual(selected_sites, [1]) # Expected selected site numbers

    def test_8(self): 
        d = 1000
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 100) # Expected total annual revenue
        self.assertEqual(selected_sites, [7]) # Expected selected site numbers


    def test_9(self): 
        d = 1
        site_list =  [50, -10, -12, -65, -40, -95, -100, -12, -20, 30] 
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 80) # Expected total annual revenue
        self.assertEqual(selected_sites, [1,10]) # Expected selected site numbers    


    def test_given_examples(self):
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        self.assertEqual(restaurantFinder(1, site_list), (252, [1, 4, 6, 8, 10])) # test all positive, d < len(site_list)
        self.assertEqual(restaurantFinder(2, site_list), (245, [1, 4, 7, 10]))
        self.assertEqual(restaurantFinder(3, site_list), (175, [1, 6, 10]))
        self.assertEqual(restaurantFinder(7, site_list), (100, [7]))  # test all positive, pick 1 even with enough room for 2
        self.assertEqual(restaurantFinder(0, site_list), (434, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) # test all pos, d = 0

    def test_mine(self):        
        self.assertEqual(restaurantFinder(1, [1000, 900, 1000, 2000]), (3000, [1, 4])) # test anti-greedy
        self.assertEqual(restaurantFinder(10, [100, 1, 1000]), (1000, [3])) # test all positive, d > len(site_list)

    def my_actual_test(self):
        self.assertEqual(restaurantFinder(1, [2,1,3,9,1]), (10, [2, 4])) # test anti-greedy

    
    def test_01(self):

        # initialising test
        d = 0
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 434)
        self.assertEqual(selected_sites, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_02(self):

        # initialising test
        d = 1
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 252)
        self.assertEqual(selected_sites, [1, 4, 6, 8, 10])

    def test_03(self):

        # initialising test
        d = 2
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 245)
        self.assertEqual(selected_sites, [1, 4, 7, 10])

    def test_04(self):

        # initialising test
        d = 3
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 175)
        self.assertEqual(selected_sites, [1, 6, 10])

    def test_05(self):

        # initialising test
        d = 7
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 100)
        self.assertEqual(selected_sites, [7])
    
    def test_06(self):
        
        # initialising test
        d = 100
        site_list = [50, 10, 12, 65, 40, 95, 100, 12, 20, 30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 100)
        self.assertEqual(selected_sites, [7])

    def test_07(self):
        
        # initialising test
        d = 1
        site_list = [1000, 900, 1000, 2000]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 3000)
        self.assertEqual(selected_sites, [1, 4])

    def test_08(self):
        
        # initialising test
        d = 10
        site_list = [100, 1, 1000]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 1000)
        self.assertEqual(selected_sites, [3])
    
    def test_09(self):
        
        # initialising test
        d = 1
        site_list = [100, 1, 100, 100, 100]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 300)
        self.assertEqual(selected_sites, [1, 3, 5])

    def test_10(self):
        
        # initialising test
        d = 10
        site_list = [100, 1, 1000, 100]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 1000)
        self.assertEqual(selected_sites, [3])
    
    def test_11(self):
        
        # initialising test
        d = 1
        site_list = [1, 1000, 1]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 1000)
        self.assertEqual(selected_sites, [2])

    def test_12(self):
        
        # initialising test
        d = 1
        site_list = [1, 4, 3, 4, 3]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 8)
        self.assertEqual(selected_sites, [2, 4])
    
    def test_13(self):
        
        # initialising test
        d = 10
        site_list = [100, 0, 2, 3, 0, 200, 0, 0, 3, 4, 2, 4, 50]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 200)
        self.assertEqual(selected_sites, [6])

    def test_14(self):
        
        # initialising test
        d = 10
        site_list = [30]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 30)
        self.assertEqual(selected_sites, [1])
    
    def test_15(self):
        
        # initialising test
        d =  2
        site_list =  [50, 100, 50, 100, 50, 100]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 200)
        self.assertEqual(selected_sites, [2, 6])

    def test_16(self):
        
        # initialising test
        d = 0
        site_list = [0, 0, 0]
        total_revenue, selected_sites = restaurantFinder(d, site_list)

        # testing
        self.assertEqual(total_revenue, 0)
        self.assertIn(selected_sites, [[1, 2, 3], [], [1]])

    def test_17(self):
        d = 1
        site_list = [4,3,2,7]
        total_revenue, selected_sites = restaurantFinder(d, site_list)
        self.assertEqual(total_revenue, 11)
        self.assertEqual(selected_sites, [1, 4])

    def test_18(self):
        site_list = [5,3,2,10,7,12,1]
        self.assertEqual(restaurantFinder(1, site_list), (27, [1, 4, 6])) 

    def test_19(self):
        site_list = [5,3,2,10,7,12,1]
        self.assertEqual(restaurantFinder(2, site_list), (17, [1, 6]))

    def test_20(self):
        site_list = [5,3,2,10,7,12,1]
        self.assertEqual(restaurantFinder(3, site_list), (17, [1, 6]))

    def test_21(self):
        site_list = [5,3,2,10,7,12,1]
        self.assertEqual(restaurantFinder(4, site_list), (12, [6]))






class TestClimbKing(unittest.TestCase):

    def test_1(self):
        paths = [(0, 1, 4), (0, 3, 2), (0, 2, 3), (2, 3, 2), (3, 0, 3)]
        keys = [(0, 5), (3, 2), (1, 3)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [1, 2]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 7)
        self.assertEqual(route, [0, 1])

    def test_02(self):

        # initialising test
        paths = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2), (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2), (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
        keys = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
        graph = FloorGraph(paths, keys)
        start = 1
        exits = [7, 2, 4]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 9)
        self.assertEqual(route, [1, 7])

    def test_03(self):

        # initialising test
        paths = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2), (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2), (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
        keys = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
        graph = FloorGraph(paths, keys)
        start = 7
        exits = [8]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 6)
        self.assertEqual(route, [7, 8])

    def test_04(self):

        # initialising test
        paths = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2), (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2), (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
        keys = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
        graph = FloorGraph(paths, keys)
        start = 1
        exits = [3, 4]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 10)
        self.assertEqual(route, [1, 5, 6, 3])

    def test_05(self):

        # initialising test
        paths = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2), (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2), (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
        keys = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
        graph = FloorGraph(paths, keys)
        start = 1
        exits = [0, 4]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 11)
        self.assertEqual(route, [1, 5, 6, 4])

    def test_06(self):

        # initialising test
        paths = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2), (5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2), (8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]
        keys = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
        graph = FloorGraph(paths, keys)
        start = 3
        exits = [4]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 20)
        self.assertEqual(route, [3, 4, 8, 7, 3, 4])
    
    def test_07(self):

        # initialising test
        paths = [(0, 1, 4), (0, 3, 2), (0, 2, 3), (2, 3, 2), (3, 0, 3)]
        keys = [(0, 5), (3, 2), (1, 3)]
        graph = FloorGraph(paths, keys)
        start = 1
        exits = [2]
        outcome = graph.climb(start, exits)

        # testing
        self.assertEqual(outcome, None)

    def test_08(self):

        # initialising test
        paths = [(0, 1, 4), (0, 2, 3), (1, 0, 2), (1, 3, 3), (3, 2, 3)]
        keys = [(1, 10)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [2]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 19)
        self.assertEqual(route, [0, 1, 0, 2])
    
    def test_09(self):

        # initialising test
        paths = [(0, 1, 5)]
        keys = [(0, 5)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [1]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 10)
        self.assertEqual(route, [0, 1])

    def test_10(self):

        # initialising test
        paths = [(0, 1, 5)]
        keys = [(0, 5)]
        graph = FloorGraph(paths, keys)
        start = 1
        exits = [0, 1]
        outcome = graph.climb(start, exits)

        # testing
        self.assertEqual(outcome, None)

    def test_11(self):

        # initialising test
        paths = [(0, 1, 3), (0, 3, 10), (0, 4, 5), (1, 2, 2), (2, 3, 4), (3, 0, 10), (3, 4, 1), (4, 0, 3)]
        keys = [(0, 10), (3, 1)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [3]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 10)
        self.assertEqual(route, [0, 1, 2, 3])

    def test_12(self):

        # initialising test
        paths = [(0, 1, 3), (0, 3, 10), (0, 4, 5), (1, 2, 2), (2, 3, 4), (3, 0, 10), (3, 4, 1), (4, 0, 3)]
        keys = [(0, 10), (3, 1)]
        graph = FloorGraph(paths, keys)
        start = 4
        exits = [4]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 14)
        self.assertEqual(route, [4, 0, 1, 2, 3, 4])

    def test_13(self):

        # initialising test
        paths = [(0, 1, 3), (0, 3, 10), (0, 4, 5), (1, 2, 2), (2, 3, 4), (3, 0, 10), (3, 4, 1), (4, 0, 3)]
        keys = [(0, 10), (3, 1)]
        graph = FloorGraph(paths, keys)
        start = 3
        exits = [0]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 5)
        self.assertEqual(route, [3, 4, 0])

    def test_14(self):

        # initialising test
        paths = [(0, 1, 3), (0, 3, 10), (0, 4, 5), (1, 2, 2), (2, 3, 4), (3, 0, 10), (3, 4, 1), (4, 0, 3)]
        keys = [(0, 10), (3, 1)]
        graph = FloorGraph(paths, keys)
        start = 4
        exits = [1, 3]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 13)
        self.assertEqual(route, [4, 0, 1, 2, 3])

    def test_15(self):

        # initialising test
        paths = [(0, 1, 3), (0, 3, 10), (0, 4, 5), (1, 2, 2), (2, 3, 4), (3, 0, 10), (3, 4, 1), (4, 0, 3)]
        keys = [(0, 10), (3, 1)]
        graph = FloorGraph(paths, keys)
        start = 1
        exits = [0, 4]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 8)
        self.assertEqual(route, [1, 2, 3, 4])

    def test_16(self):
         # initialising test
        paths = [(0, 1, 3), (0, 3, 10), (0, 4, 5), (1, 2, 2), (2, 3, 4), (3, 0, 10), (3, 4, 1), (4, 0, 3)]
        keys = [(0, 10), (3, 1)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [0, 1, 2, 3, 4]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 10)
        self.assertIn(route, [[0], [0, 1, 2, 3]])

    def test_17(self):
        paths = [(0, 1, 3), (1, 2, 3), (2, 3, 1), (0, 3, 2)]
        keys = [(3, 1)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [0]
        outcome = graph.climb(start, exits)

        self.assertEqual(outcome, None)

    def test_18(self):
        paths = [(0,1,2), (1,2,2), (2,3,2), (3,2,2), (2,1,2), (1,0,2)]
        keys = [(3,5)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [0]
        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 17)
        self.assertEqual(route, [0, 1, 2, 3, 2, 1, 0])

    def test_19(self):

        # initialising test
        paths = [(0,1,1), (1,0,1), (1,2,1), (2,3,1), (3,4,1), (4,1,1)]
        keys = [(0,2)]
        graph = FloorGraph(paths, keys)
        start = 2
        exits = [3]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 9)
        self.assertEqual(route, [2,3,4,1,0,1,2,3])

    def test_20(self):

        # initialising test
        paths = [(0,1,1), (1,0,1), (1,2,1), (2,3,1), (3,4,1), (4,1,1)]
        keys = [(0,2)]
        graph = FloorGraph(paths, keys)
        start = 2
        exits = [3]
        total_time, route = graph.climb(start, exits)

        # testing
        self.assertEqual(total_time, 9)
        self.assertEqual(route, [2,3,4,1,0,1,2,3])

    def test_21(self):
        # initialising test
        paths = [(0,1,0), (1,2,0), (2,3,0), (3,2,0), (1,0,0), (3,1,0)]
        keys = [(3,0)]
        graph = FloorGraph(paths, keys)
        start = 0
        exits = [0]
        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 0)
        self.assertEqual(route, [0, 1, 2, 3, 1, 0])

    def test_22(self):
        # initialising test
        paths = [(0,1,0), (1,2,0), (0,2,4), (2,0,0)]
        keys = [(2,3)]
        graph = FloorGraph(paths, keys)

        start = 0
        exits = [0]
        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 3)
        self.assertEqual(route, [0, 1, 2, 0])

    def test_23(self):
        # initialising test
        paths = [(0,1,1), (1,2,2), (1,3,2)]
        keys = [(2,3), (3,0)]
        graph = FloorGraph(paths, keys)

        start = 0
        exits = [2]
        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 6)
        self.assertEqual(route, [0, 1, 2])

    def test_24(self):
        # initialising test
        paths = [(0,1,1), (1,2,2), (1,3,2)]
        keys = [(2,4), (3,4)]
        graph = FloorGraph(paths, keys)

        start = 0
        exits = [1]
        outcome = graph.climb(start, exits)

        self.assertEqual(outcome, None)

    def test_25(self):
        # initialising test
        paths = [(0,0,0)]
        keys = [(0,4)]
        graph = FloorGraph(paths, keys)

        start = 0
        exits = [0]
        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 4)
        self.assertEqual(route, [0])

    def test_26(self):
        paths = [(0,1,0), (1,2,0), (2,3,0), (3,4,0)]
        keys = [(0,5), (1,5), (2,3), (3,2), (4,1)]
        graph = FloorGraph(paths, keys)
    
        start = 0
        exits = [0,1,2,3,4]

        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 1)
        self.assertEqual(route, [0,1,2,3,4])

    def test_27(self):
        paths = [(0,1,3), (1,2,4), (2,0,0), (2,1,7)]
        keys = [(2,1)]

        graph = FloorGraph(paths, keys)

        start = 0
        exits = [1]
        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 11)
        self.assertEqual(route, [0,1,2,0,1])

    def test_28(self):
        paths = [(0,1,0), (0,2,0), (2,1,2), (1,2,2)]
        keys = [(1,2), (2,1)]

        graph = FloorGraph(paths, keys)

        start = 0 
        exits = [0]

        outcome = graph.climb(start, exits)
        self.assertEqual(outcome, None)

    def test_29(self):
        paths = [(0,1,5), (1,2,0), (2,0,0), (1,0,2)]
        keys = [(1,3)]

        graph = FloorGraph(paths, keys)

        start = 0
        exits = [0]

        total_time, route = graph.climb(start, exits)

        self.assertEqual(total_time, 8)
        self.assertEqual(route, [0,1,2,0])

    def test_30(self):
        paths = [(1,0,0)]
        keys = [(0,0)]
        graph = FloorGraph(paths, keys)

        start = 0 
        exits = [1]

        outcome = graph.climb(start, exits)
        self.assertEqual(outcome, None)





    
        
if __name__ == '__main__':

 unittest.main()